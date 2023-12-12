"""The pre-trained audio VQAGAN which tokenizes spectograms"""
import functools
from typing import Any, Callable, Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax

from t5x.examples.unified_io.layers import combine_masks, dynamic_vector_slice_in_dim, \
  combine_biases

from t5x.examples.unified_io import layers
from t5x.examples.unified_io.config import ImageViTVQGANConfig

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]
Dtype = Any

default_kernel_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal')

default_bias_init = nn.initializers.zeros

ACT2FN = {
    "tanh": nn.tanh,
    "relu": nn.relu,
    "swish": nn.swish,
}


def l2_normalize(x, axis=None, eps=1e-12):
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.0
  act_fn: str = 'relu'
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    
    x = layers.DenseGeneral(
        features=self.mlp_dim,
        dtype=self.dtype,
        use_bias=True,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('embed', 'mlp'),
        bias_axes=('mlp',),
        )(  # pytype: disable=wrong-arg-types
            inputs)

    x = ACT2FN[self.act_fn](x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

    output = layers.DenseGeneral(
        features=actual_out_dim,
        dtype=self.dtype,
        use_bias=True,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('mlp', 'embed'),
        bias_axes=('embed',)
        )(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class MultiHeadDotProductAttention(nn.Module):
  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  dropout_broadcast_dims: Sequence[int] = ()
  kernel_init: Initializer = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal')
  float32_logits: bool = False  # computes logits in float32 for stability.

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               abs_bias: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> Array:
    projection = functools.partial(
      layers.DenseGeneral,
      axis=-1,
      features=(self.num_heads, self.head_dim),
      kernel_axes=('embed', 'joined_kv'),
      dtype=self.dtype)

    query = projection(kernel_init=self.kernel_init, name='query')(inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    if decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      cache_mask = self.variable('cache', 'cache_mask', jnp.zeros,
                                 (query.shape[0], 1, 1, query.shape[1]), jnp.float32)
      if is_initialized:
        batch, num_heads, head_dim, length = (cached_key.value.shape)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # Sanity shape check of cached key against input query.
        expected_shape = (batch, 1, num_heads, head_dim)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))

        # Create a OHE of the current index. NOTE: the index is increased below.
        cur_index = cache_index.value
        one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
        # In order to update the key, value caches with the current key and
        # value, we move the length axis to the back, similar to what we did for
        # the cached ones above.
        # Note these are currently the key and value of a single position, since
        # we feed one position at a time.
        one_token_key = jnp.moveaxis(key, -3, -1)
        one_token_value = jnp.moveaxis(value, -3, -1)
        # Update key, value caches with our new 1d spatial slices.
        # We implement an efficient scatter into the cache via one-hot
        # broadcast and addition.
        key = cached_key.value + one_token_key * one_hot_indices
        value = cached_value.value + one_token_value * one_hot_indices
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # Move the keys and values back to their original shapes.
        key = jnp.moveaxis(key, -1, -3)
        value = jnp.moveaxis(value, -1, -3)

        # Causal mask for cached decoder self-attention: our single query
        # position should only attend to those key positions that have already
        # been generated and cached, not the remaining zero elements.
        # mask = jnp.logical_or(cache_mask.value, mask).astype(jnp.int32)
        # cache_mask.value = mask

        # if cur_index == 20:
        # import ipdb; ipdb.set_trace()

        mask = (cache_mask.value + mask * one_hot_indices).astype(jnp.float32)
        cache_mask.value = mask

        mask = combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(length) <= cur_index,
            # (1, 1, length) represent (head dim, query length, key length)
            # query length is 1 because during decoding we deal with one
            # index.
            # The same mask is applied to all batch elements and heads.
            (batch, 1, 1, length)))


        # Grab the correct relative attention bias during decoding. This is
        # only required during single step decoding.
        if bias is not None:
          # The bias is a full attention matrix, but during decoding we only
          # have to take a slice of it.
          # This is equivalent to bias[..., cur_index:cur_index+1, :].
          bias = dynamic_vector_slice_in_dim(
            jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

          abs_bias = dynamic_vector_slice_in_dim(
            jnp.squeeze(abs_bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
        mask > 0,
        jnp.full(mask.shape, 0.).astype(self.dtype),
        jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias, abs_bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = layers.dot_product_attention(
      query,
      key,
      value,
      bias=attention_bias,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      dropout_broadcast_dims=self.dropout_broadcast_dims,
      deterministic=deterministic,
      dtype=self.dtype,
      float32_logits=self.float32_logits)

    out_kernel_init = self.kernel_init

    # Back to the original inputs dimensions.
    out = layers.DenseGeneral(
      features=inputs_q.shape[-1],  # output dim is set to the input dim.
      axis=(-2, -1),
      kernel_init= out_kernel_init,
      kernel_axes=('joined_kv', 'embed'),
      dtype=self.dtype,
      name='out')(
      x)

    return out


class TransformerLayer(nn.Module):
  mlp_dim: int
  num_heads: int
  head_dim: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  use_bias: bool = False
  act_fn: str = 'relu'
  float32_attention_logits: bool = False

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = layers.LayerNormWithBias(
        dtype=self.dtype,
        bias_init=nn.initializers.zeros,
        scale_init=nn.initializers.ones,
    )(inputs)

    x = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        float32_logits=self.float32_attention_logits
    )(x, x)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = layers.DropPath(rate=self.droppath_rate)(x, deterministic=deterministic) + inputs
    
    y = layers.LayerNormWithBias(
        dtype=self.dtype,
        bias_init=nn.initializers.zeros,
        scale_init=nn.initializers.ones,
    )(x)

    # MLP block.
    y = MlpBlock(
        mlp_dim=self.mlp_dim, 
        dtype=self.dtype, 
        act_fn=self.act_fn,
        dropout_rate=self.dropout_rate
    )(y, deterministic=deterministic)
    return x + layers.DropPath(rate=self.droppath_rate)(y, deterministic=deterministic)


class Transformer(nn.Module):
  num_layers: int
  mlp_dim: int
  num_heads: int
  head_dim: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  add_position_embedding: bool = True
  use_bias: bool = False
  act_fn: str = 'relu'

  @nn.compact
  def __call__(self, x, *, train):

    assert x.ndim == 3  # (batch, len, emb)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    dpr = [x for x in np.linspace(0, self.droppath_rate, self.num_layers)]
    for lyr in range(self.num_layers):
      x = TransformerLayer(
          mlp_dim=self.mlp_dim,
          head_dim=self.head_dim,
          dropout_rate=self.dropout_rate,
          dtype=self.dtype,
          droppath_rate=dpr[lyr],
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads,
          use_bias=self.use_bias,
          act_fn=self.act_fn
      )(x, deterministic=not train)

    x = layers.LayerNormWithBias(
        bias_init=nn.initializers.zeros,
        scale_init=nn.initializers.ones,
        name='encoder_norm')(x)  

    return x


class VectorQuantizer(nn.Module):
  n_e: int
  e_dim: int
  beta: float = 0.25
  embedding_init: Callable[[PRNGKey, Shape, DType], Array] = jax.nn.initializers.uniform(2.0)
  dtype: Any = jnp.float32

  def setup(self):
    self.embedding = param_with_axes(
        'embedding',
        self.embedding_init, (self.n_e, self.e_dim),
        jnp.float32,
        axes=(('vocab', 'embed')))

  def get_codebook_entry(self, indices):
    # indices are expected to be of shape (batch, num_tokens)
    # get quantized latent vectors
    z_q = jnp.take(self.embedding, indices, axis=0)
    # normalize latent variable (Ze(x) in the paper)
    z_q = l2_normalize(z_q, axis=-1)
    return z_q
    
  @nn.compact
  def __call__(self, z: Array) -> Array:

    z_reshaped = jnp.reshape(z, (-1, self.e_dim))
    # first normalize the input.
    z_reshaped_norm = l2_normalize(z_reshaped, axis=-1) #/ jnp.linalg.norm(z_reshaped, axis=-1, keepdims=True)
    embedding_norm = l2_normalize(self.embedding, axis=-1) #/ jnp.linalg.norm(self.embedding, axis=-1, keepdims=True)

    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = jnp.sum(z_reshaped_norm ** 2, axis=1, keepdims=True) + \
        jnp.sum(embedding_norm ** 2, axis=1) - 2 * \
        jnp.einsum('ij,kj->ik', z_reshaped_norm, embedding_norm)

    min_encoding_indices = jnp.reshape(jnp.argmin(d, axis=1), z.shape[:-1])

    # z_q = jnp.take(self.embedding, min_encoding_indices, axis=0)
    z_q = self.get_codebook_entry(min_encoding_indices)
    z_norm = l2_normalize(z, axis=-1)

    # e_mean = jnp.mean(min_encoding_indices, axis=0)
    # perplexity = jnp.exp(-jnp.sum(e_mean * jnp.log(e_mean + 1e-10)))
    perplexity = None
    min_encodings = None

    loss = self.beta * jnp.mean(jnp.square((jax.lax.stop_gradient(z_q)-z_norm))) + \
            jnp.mean(jnp.square((z_q - jax.lax.stop_gradient(z_norm))))

    z_q = z + jax.lax.stop_gradient(z_q - z)

    return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


class ViTEncoder(nn.Module):
  config: ImageViTVQGANConfig

  def setup(self):
    cfg = self.config
    self.encoder_position_embedding = layers.get_2d_sincos_pos_embed(
      emb_dim=cfg.encoder_hidden_size,
      image_size=cfg.default_input_size,
      image_patch_size=cfg.patch_size,
      dtype=cfg.dtype,
      class_token=False,
    )

  @nn.compact
  def __call__(self, x, train=False):
    cfg = self.config
    x = layers.space_to_depth(x, spatial_block_size=cfg.patch_size[0])
    
    x = layers.DenseGeneral(
        cfg.encoder_hidden_size,
        dtype=cfg.dtype,
        use_bias=True,
        kernel_init = default_kernel_init,
        bias_init = default_bias_init,
        kernel_axes=('image_patch', 'embed'),
        bias_axes=('embed',),
        name='embedding',
    )(x)

    x += jnp.expand_dims(self.encoder_position_embedding, 0)

    x = Transformer(
        num_layers=cfg.encoder_num_layers,
        mlp_dim=cfg.encoder_mlp_dim,
        num_heads=cfg.encoder_num_heads,
        head_dim=cfg.encoder_head_dim,
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        droppath_rate=cfg.droppath_rate,
        attention_dropout_rate=cfg.attention_dropout_rate,
        add_position_embedding=cfg.add_position_embedding,
        use_bias=cfg.use_bias,
        act_fn=cfg.act_fn,
        )(x, train=train)

    x = ACT2FN[cfg.act_fn](x)

    x = layers.DenseGeneral(
        features=cfg.proj_dim,
        dtype=cfg.dtype,
        use_bias=cfg.use_bias,
        kernel_init=default_kernel_init,
        kernel_axes=('embed', 'mlp'),
        bias_axes=('mlp',),
        name='encoder_proj'
        )(x)


    x = layers.LayerNormWithBias(
        use_scale = False,
        dtype=cfg.dtype,
        name='encoder_norm')(x)

    return x


class ViTDecoder(nn.Module):
  config: ImageViTVQGANConfig

  def setup(self):
    cfg = self.config
    self.decoder_position_embedding = layers.get_2d_sincos_pos_embed(
      emb_dim=cfg.decoder_hidden_size,
      image_size=cfg.default_input_size,
      image_patch_size=cfg.patch_size,
      dtype=cfg.dtype,
      class_token=False,
    )

  @nn.compact
  def __call__(self, x, train=False):
    cfg = self.config
    x = layers.DenseGeneral(
        cfg.decoder_hidden_size,
        dtype=cfg.dtype,
        use_bias=cfg.use_bias,
        kernel_init=default_kernel_init,
        kernel_axes=('image_patch', 'embed'),
        name='decoder_proj',
        )(x)

    x += jnp.expand_dims(self.decoder_position_embedding, 0)
    
    x = Transformer(
        num_layers=cfg.decoder_num_layers,
        mlp_dim=cfg.decoder_mlp_dim,
        head_dim=cfg.decoder_head_dim,
        num_heads=cfg.decoder_num_heads,
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        droppath_rate=cfg.droppath_rate,
        attention_dropout_rate=cfg.attention_dropout_rate,
        add_position_embedding=cfg.add_position_embedding,
        use_bias=cfg.use_bias,
        act_fn=cfg.act_fn,
        )(x, train=train)

    img_size = cfg.default_input_size
    x = jnp.reshape(x, (-1, img_size[0] // cfg.patch_size[0], img_size[1] // cfg.patch_size[1], cfg.decoder_hidden_size))

    x = layers.ConvTranspose(
        features=cfg.output_channel, 
        kernel_size=cfg.patch_size, 
        strides=cfg.patch_size,
        use_bias=cfg.use_bias,
        kernel_init=default_kernel_init,
        kernel_axes=('axis_0', 'axis_1', 'embed', 'axis_3'),
        bias_axes=('axis_3',),
        )(x)
    
    return x


class ASTVQGAN(nn.Module):
  """ViTVQAGAN implementation compatible with https://arxiv.org/abs/2104.01778"""
  config: ImageViTVQGANConfig

  def setup(self):
    cfg = self.config
    self.quantize = VectorQuantizer(
        n_e=cfg.vocab_size,
        e_dim=cfg.proj_dim,
        beta=0.25
    )
    self.encoder = ViTEncoder(cfg)
    self.decoder = ViTDecoder(cfg)

  def encode(self, x, train=True):
    return self.encoder(x, train)

  def decode(self, x, train=True):
    return self.decoder(x, train)

  def get_quantize_from_emb(self, h):
    z, _, [_, _, indices] = self.quantize(h)
    return indices

  def decode_code(self, code_b):
    quant_b = self.quantize.get_codebook_entry(code_b)
    dec = self.decode(quant_b)
    return dec
    
  def get_codebook_indices(self, x, vae_decode=False, train=False):
    h = self.encode(x, train=train)
    z, _, [_, _, indices] = self.quantize(h)
    if vae_decode:
        # So the decoding parameters are registered during initialization
        dec = self.decode(z, train)

    return jnp.reshape(indices, (jnp.shape(h)[0], -1))

  @nn.compact
  def __call__(self, x, training=False):
    cfg = self.config
    h = self.encode(x, training)
    z, _, [_, _, indices] = self.quantize(h)
    if cfg.use_decoder:
      # This method is only used during initialization, so it should call decode so those
      # parameters also get initialized
      dec = self.decode(z, training)
    else:
      dec = None
    return z, dec
