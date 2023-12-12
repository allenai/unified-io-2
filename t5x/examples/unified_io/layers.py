"""UnifiedIO 2 layers, modified from t5x.layers"""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
import einops
from flax import linen
import math

from jax.nn import initializers
from flax.linen.module import Module, compact, merge_param

default_kernel_init = nn.initializers.lecun_normal()
default_init = nn.initializers.lecun_normal()
zero_init =  nn.initializers.zeros
one_init = nn.initializers.ones

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
Axes = Union[int, Iterable[int]]

# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)

def reverse_space_to_depth(
    frames: jnp.ndarray,
    temporal_block_size: int = 1,
    spatial_block_size: int = 1) -> jnp.ndarray:
  """Reverse space to depth transform."""
  if len(frames.shape) == 4:
    return einops.rearrange(
        frames, 'b h w (dh dw c) -> b (h dh) (w dw) c',
        dh=spatial_block_size, dw=spatial_block_size)
  elif len(frames.shape) == 5:
    return einops.rearrange(
        frames, 'b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c',
        dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
  else:
    raise ValueError(
        'Frames should be of rank 4 (batch, height, width, channels)'
        ' or rank 5 (batch, time, height, width, channels)')

def space_to_depth(
    frames: jnp.ndarray,
    temporal_block_size: int = 1,
    spatial_block_size: int = 1) -> jnp.ndarray:
  """Space to depth transform."""
  if len(frames.shape) == 4:
    return einops.rearrange(
        frames, 'b (h dh) (w dw) c -> b (h w) (dh dw c)',
        dh=spatial_block_size, dw=spatial_block_size)
  elif len(frames.shape) == 5:
    return einops.rearrange(
        frames, 'b (t dt) (h dh) (w dw) c -> b t (h w) (dt dh dw c)',
        dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
  else:
    raise ValueError(
        'Frames should be of rank 4 (batch, height, width, channels)'
        ' or rank 5 (batch, time, height, width, channels)')


def l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm.
    This specialized function exists for numerical stability reasons.
    Args:
      x: An input ndarray.
      axis: Dimension along which to normalize, e.g. `1` to separately normalize
        vectors in a batch. Passing `None` views `t` as a flattened vector when
        calculating the norm (equivalent to Frobenius norm).
      eps: Epsilon to avoid dividing by zero.
    Returns:
      An array of the same shape as 'x' L2-normalized along 'axis'.
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          dropout_broadcast_dims: Sequence[int] = (-2, ),
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False,
                          depth_normalize=True,
                          clip_attn_logit=None,
                          logit_scale=None,
                          logit_scale_max=math.log(1. / 0.01),
                          ):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  if logit_scale is not None:
    attn_weights = jnp.einsum('bqhd,bkhd->bhqk', l2_normalize(query, -1), l2_normalize(key, -1))
    if logit_scale_max is not None:
        logit_scale = jnp.exp(jnp.clip(logit_scale, a_max=logit_scale_max))
    else:
        logit_scale = jnp.exp(logit_scale)
    attn_weights = attn_weights * logit_scale

  else:
    # calculate attention matrix
    if depth_normalize:
      depth = query.shape[-1]
      query = query / jnp.sqrt(depth).astype(dtype)

    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

    # clip attention weight 
    if clip_attn_logit:
      attn_weights = jnp.clip(attn_weights, -clip_attn_logit, clip_attn_logit)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)
  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    for dim in dropout_broadcast_dims:
      dropout_shape[dim] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))

dynamic_vector_slice_in_dim_2 = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(0, None, None, None))


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """

  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  use_bias: bool = False
  dropout_rate: float = 0.
  dropout_broadcast_dims: Sequence[int] = (-2, )
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal')
  float32_logits: bool = True  # computes logits in float32 for stability.
  qk_norm: bool = True
  use_head_scale: bool = False
  depth_normalize: bool = True
  clip_attn_logit: Any = None
  scaled_cosine: bool = False

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               abs_bias: Optional[Array] = None,
               q_sinusoids: Optional[Array] = None,
               k_sinusoids: Optional[Array] = None,
               attn_pattern_mask: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      q_sinusoids: sinusoidal values for the block diagonal matrix of query RoPE.
        `[batch, q_length, 2 (cos then sin) * rotary_hsize <= size_per_head]`.
      k_sinusoids: sinusoidal values for the block diagonal matrix of key RoPE.
        `[batch, kv_length, 2 (cos then sin) * rotary_hsize <= size_per_head]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        use_bias=self.use_bias,
        kernel_axes=('embed', 'joined_kv'),
        bias_axes=('joined_kv',),
        dtype=self.dtype)

    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    value_init = self.kernel_init

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=self.kernel_init, name='query')(inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection(kernel_init=value_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    if self.qk_norm:
      query = LayerNorm(dtype=self.dtype, name='query_norm')(query)
      key = LayerNorm(dtype=self.dtype, name='key_norm')(key)

    if q_sinusoids is not None:
      q_sinusoids = with_sharding_constraint(q_sinusoids, ('batch', 'length', 'kv'))
      query = apply_rotary(query, q_sinusoids)
    if k_sinusoids is not None:
      k_sinusoids = with_sharding_constraint(k_sinusoids, ('batch', 'length', 'kv'))
      key = apply_rotary(key, k_sinusoids)

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
      # cache_mask = self.variable('cache', 'cache_mask', jnp.zeros, 
      #                             (query.shape[0], 1, 1, query.shape[1]), jnp.float32)
      if is_initialized:
        batch, num_heads, head_dim, length = (cached_key.value.shape)
        if bias is not None:
          if length != bias.shape[-1]:
            raise ValueError(f"Length was {length}, but bias length is {bias.shape[-1]}. "
                             f"This probably means the cache was initialized to the incorrect "
                             f"length")
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

        # mask = (cache_mask.value + mask * one_hot_indices).astype(jnp.float32)
        # cache_mask.value = mask
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

        if attn_pattern_mask is not None:
          attn_pattern_mask = dynamic_vector_slice_in_dim(attn_pattern_mask, jnp.reshape(cur_index, (-1)), 1, -2)
          attn_pattern_mask = jnp.squeeze(attn_pattern_mask, axis=0)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    if attn_pattern_mask is not None:
      pattern_bias = lax.select(
          attn_pattern_mask > 0,
          jnp.full(attn_pattern_mask.shape, 0.).astype(self.dtype),
          jnp.full(attn_pattern_mask.shape, -1e10).astype(self.dtype))
    else:
      pattern_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, pattern_bias, bias, abs_bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    if self.scaled_cosine:
      scale_init = lambda *_ : jnp.array(jnp.log(10 * jnp.ones(self.num_heads)), dtype=jnp.float32)
      logit_scale = param_with_axes(
          "logit_scale", 
          scale_init,
          axes=('heads',))
      logit_scale = jnp.reshape(logit_scale, (1,self.num_heads,1, 1))
    else:
      logit_scale = None

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        depth_normalize=self.depth_normalize,
        clip_attn_logit=self.clip_attn_logit,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        dropout_broadcast_dims=self.dropout_broadcast_dims,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits, 
        logit_scale=logit_scale)

    if self.use_head_scale:
      head_scale = param_with_axes(
          "head_scale", 
          jax.nn.initializers.ones, 
          (self.num_heads,), 
          jnp.float32, 
          axes=('heads',))
      head_scale = jnp.reshape(head_scale, (1,self.num_heads,1, 1))
      x = x * head_scale
    
    out_init = self.kernel_init
    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        use_bias=self.use_bias,
        kernel_init=out_init,
        bias_init=nn.initializers.zeros,
        kernel_axes=('joined_kv', 'embed'),
        bias_axes=('embed',),
        dtype=self.dtype,
        name='out')(
            x)
    return out


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

#------------------------------------------------------------------------------
# Convolution layers
#------------------------------------------------------------------------------

EPS = 1e-17
NEG_INF = -1e30

def gumbel_sample(rng, shape):
  """Sample Gumbel noise."""
  uniform = random.uniform(rng, shape=shape)
  return -jnp.log(EPS - jnp.log(uniform + EPS))
  
def gumbel_softmax_sample(rng, logits, tau=1., dim=-1, hard=True):
  """Sample from the Gumbel softmax / concrete distribution."""  
  gumbel_noise = random.gumbel(rng, logits.shape)

  y_soft = nn.softmax((logits + gumbel_noise) / tau, axis=-1)

  # if hard:
  #   index = jnp.argmax(y_soft, axis=dim)
  #   y_hard = common_utils.onehot(index, logits.shape[-1])
  #   ret = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
  # else:
  #   ret = y_soft
  return jnp.argmax(y_soft, axis=dim)

class GumbelQuantize(nn.Module):
  n_e: int
  e_dim: int
  # beta: float = 0.25
  embedding_init: Initializer = default_embed_init
  dtype: Any = jnp.float32
  params_init: Any = None

  def setup(self):

    w_init = default_kernel_init if self.params_init is None else lambda *_ : jnp.transpose(jnp.array(self.params_init['proj']['weight']), (2, 3, 1, 0))
    b_init = zero_init if self.params_init is None else lambda *_ : jnp.array(self.params_init['proj']['bias']) 

    self.proj = Conv(
      features=self.n_e,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='proj')

    kernel_init = self.embedding_init if self.params_init is None \
        else lambda *_ : jnp.array(self.params_init['embed']['weight'], dtype=jnp.float32)

    self.embedding = param_with_axes(
        'embed',
        kernel_init, (self.n_e, self.e_dim),
        jnp.float32,
        axes=(('vocab', 'embed')))

  def get_codebook_entry(self, indices):
    min_encodings = jax.nn.one_hot(indices, self.n_e, dtype=self.dtype)
    z_q = jnp.einsum('bqk,kd->bqd', min_encodings, self.embedding)
    return z_q

  @nn.compact
  def __call__(self, z: Array, ) -> Array:
    
    
    logits = self.proj(z)
    
    # if self.is_initializing():
    rng = random.PRNGKey(0)
    # else:
      # rng = self.make_rng('gumbel_rng') 
    min_encoding_indices = gumbel_softmax_sample(rng, logits)

    z_q = jnp.asarray(self.embedding, self.dtype)[min_encoding_indices]

    return z_q, None, (None, None, min_encoding_indices)


class VectorQuantizer(nn.Module):
  n_e: int
  e_dim: int
  beta: float = 0.25
  embedding_init: Initializer = default_embed_init
  dtype: Any = jnp.float32
  params_init: Any = None

  def setup(self):
    kernel_init = self.embedding_init if self.params_init is None \
        else lambda *_ : jnp.array(self.params_init['embedding']['weight'], dtype=jnp.float32)

    self.embedding = param_with_axes(
        'embedding',
        kernel_init, (self.n_e, self.e_dim),
        jnp.float32,
        axes=(('vocab', 'embed')))

  def get_codebook_entry(self, indices):
    min_encodings = jax.nn.one_hot(indices, self.n_e, dtype=self.dtype)
    z_q = jnp.einsum('bqk,kd->bqd', min_encodings, self.embedding)
    return z_q

  @nn.compact
  def __call__(self, z: Array) -> Array:

    z_flattened = jnp.reshape(z, (-1, self.e_dim))
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

    d = jnp.sum(z_flattened ** 2, axis=1, keepdims=True) + \
        jnp.sum(self.embedding ** 2, axis=1) - 2 * \
        jnp.einsum('ij,kj->ik', z_flattened, self.embedding)
    
    min_encoding_indices = jnp.argmin(d, axis=1)
    z_q = jnp.asarray(self.embedding, self.dtype)[min_encoding_indices]
    z_q = jnp.reshape(z_q, z.shape)

    perplexity = None
    min_encodings = None
    loss = jnp.mean((jax.lax.stop_gradient(z_q)-z)**2) + self.beta * \
            jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)
    
    z_q = z + jax.lax.stop_gradient(z_q - z)

    return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


def nonlinearity(x):
    # swish
    return x*nn.sigmoid(x)

def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

class Conv(nn.Module):
  """Convolution Module with flexible axes.
    Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """
  features: int
  kernel_size: Iterable[int]
  strides: Union[None, int, Iterable[int]] = 1
  padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
  input_dilation: Union[None, int, Iterable[int]] = 1
  kernel_dilation: Union[None, int, Iterable[int]] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: DType = jnp.float32
  param_dtype: DType = jnp.float32
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  precision: Any = None #jax.lax.Precision('highest')
  kernel_axes: Tuple[str, ...] = ()
  bias_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a convolution to the inputs.
 
    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution
        and NDHWC for a 3D convolution. Note: this is different from the input
        convention used by `lax.conv_general_dilated`, which puts the spatial
        dimensions last.
    Returns:
      The convolved data.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    if isinstance(self.kernel_size, int):
      raise TypeError('The kernel size must be specified as a'
                      ' tuple/list of integers (eg.: [3, 3]).')
    else:
      kernel_size = tuple(self.kernel_size)
    
    def maybe_broadcast(x):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return x

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)

    strides = maybe_broadcast(self.strides)  # self.strides or (1,) * (inputs.ndim - 2)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    kernel_shape = kernel_size + (
        in_features // self.feature_group_count, self.features)

    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_shape,
        self.param_dtype,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    if self.padding == 'CIRCULAR':
      kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
      pads = [(0, 0)] + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    else:
      padding_lax = self.padding

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding_lax,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        precision=self.precision)

    if is_single_input:
      y = jnp.squeeze(y, axis=0)
    if self.use_bias:
      bias = param_with_axes(
          'bias',
          self.bias_init,
          (self.features,),
          self.param_dtype,
          axes=self.bias_axes)

      bias = jnp.asarray(bias, self.dtype)
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class AttentionPooling(nn.Module):
  n_inputs: int
  n_outputs: int
  cfg: Any

  @compact
  def __call__(self, x_in, mask, deterministic=True):
    cfg = self.cfg
    batch, seq, dim = x_in.shape
    seq = seq*self.n_inputs
    dim = dim // self.n_inputs

    x_in = jnp.reshape(x_in, (batch*seq//self.n_inputs, self.n_inputs, dim))
    x = LayerNorm(dtype=cfg.dtype, name='pre_attention_layer_norm')(x_in)

    q_in = DenseGeneral(
      features=self.n_outputs*dim,
      kernel_axes=('embed', 'mlp'),
      bias_axes=('mlp',),
      name="linear",
      dtype=cfg.dtype
    )(jnp.reshape(x_in, [-1, dim*self.n_inputs]))
    q_in = jnp.reshape(q_in, [-1, self.n_outputs, dim])
    queries = LayerNorm(dtype=cfg.dtype, name='query_norm')(q_in)

    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = MultiHeadDotProductAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      dropout_rate=cfg.dropout_rate,
      float32_logits=cfg.float32_attention_logits,
      name='attention')(
      queries, x, None, None, abs_bias=None, deterministic=deterministic)

    x = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)

    x = x + q_in

    # MLP block.
    y = LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = MlpBlock(
      intermediate_dim=cfg.mlp_dim,
      activations=cfg.mlp_activations,
      intermediate_dropout_rate=cfg.dropout_rate,
      dtype=cfg.dtype,
      name='mlp',
    )(y, deterministic=deterministic)

    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)
    y = y + x

    y = jnp.reshape(y, (batch, seq//self.n_inputs, dim))
    return y



    # Reshape tp [batch*blocks, n, dim]

#------------------------------------------------------------------------------
# DenseGeneral for attention layers.
#------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
  """A linear transformation (without bias) with flexible axes.
    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  use_bias: bool = False
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  bias_init: Initializer = nn.initializers.zeros
  kernel_axes: Tuple[str, ...] = ()
  bias_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.
    Args:
      inputs: The nd-array to be transformed.
    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                          np.prod(features))
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        jnp.float32,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))

    if self.use_bias:
      bias_shape = features
      bias_param_shape = np.prod(features)
      bias = param_with_axes(
          'bias',
          self.bias_init,
          (bias_param_shape,),
          jnp.float32,
          axes=self.bias_axes)

      bias = jnp.asarray(bias, self.dtype)
      bias = jnp.reshape(bias, bias_shape)
      y += jnp.reshape(bias, (1,) * (y.ndim - bias.ndim) + (bias.shape))

    return y


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dropout_broadcast_dims: Sequence[int] = (-2, )
  dtype: Any = jnp.float32
  use_bias: bool = False

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('linear', 'gelu') for gated-gelu.

    mlp_init = self.kernel_init
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'

      x = DenseGeneral(
          self.intermediate_dim,
          dtype=self.dtype,
          use_bias=self.use_bias,
          kernel_init=mlp_init,
          bias_init=nn.initializers.zeros,
          kernel_axes=('embed', 'mlp'),
          bias_axes=('mlp',),
          name=dense_name)(
              inputs)
              
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)

    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=self.dropout_broadcast_dims)(
            x, deterministic=deterministic)  # Broadcast along length.
    x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=mlp_init,
        bias_init=nn.initializers.zeros,
        kernel_axes=('mlp', 'embed'),
        bias_axes=('embed',),
        name='wo')(
            x)
    return output


class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init
  one_hot: bool = False
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = param_with_axes(
        'embedding',
        self.embedding_init, (self.num_embeddings, self.features),
        jnp.float32,
        axes=('vocab', 'embed'))

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
      output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class RelativePositionBiases(nn.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
  """
  num_buckets: int
  img_num_buckets: int
  max_distance: int
  img_max_distance: int
  num_heads: int
  img_width: int
  img_height: int
  dtype: Any
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init

  def setup(self) -> None:
    image_num_rel_dis = self.img_num_buckets ** 2 * 4
    self.img_relative_attention_bias = param_with_axes(
      'image_rel_embedding',
      self.embedding_init, (self.num_heads, image_num_rel_dis),
      jnp.float32,
      axes=('heads', 'relpos_buckets'))

    self.relative_attention_bias = param_with_axes(
      'rel_embedding',
      self.embedding_init, (self.num_heads, self.num_buckets),
      jnp.float32,
      axes=('heads', 'relpos_buckets'))


  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(jnp.int32) * num_buckets
      n = jnp.abs(n)
    else:
      n = jnp.maximum(n, 0)

    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        jnp.log(n.astype(jnp.float32) / max_exact + jnp.finfo(jnp.float32).eps) /
        jnp.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(jnp.int32)

    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    ret += jnp.where(is_small, n, val_if_large)
    return ret

  @staticmethod
  def _img_relative_position_bucket(relative_position_x,
                                    relative_position_y,
                                    num_buckets=8,
                                    max_distance=20):

    max_exact = num_buckets // 2                        
    nx = -relative_position_x 
    ny = -relative_position_y

    total_buckets = num_buckets ** 2
    ret = 0
    ret += (jnp.logical_and(nx <=0, ny <0)).astype(jnp.int32) * total_buckets * 3
    ret += (jnp.logical_and(nx <0, ny >=0)).astype(jnp.int32) * total_buckets * 2
    ret += (jnp.logical_and(nx >0, ny <=0)).astype(jnp.int32) * total_buckets * 1

    nx = jnp.abs(nx)
    ny = jnp.abs(ny)

    is_small_x = nx < max_exact
    val_x_if_large = max_exact + (jnp.log(nx.astype(jnp.float32) / 
        max_exact + jnp.finfo(jnp.float32).eps) / jnp.log(max_distance / 
        max_exact) * (num_buckets - max_exact)).astype(np.int32)
    
    val_x_if_large = jnp.minimum(val_x_if_large, num_buckets - 1)

    is_small_y = ny < max_exact
    val_y_if_large = max_exact + (jnp.log(ny.astype(jnp.float32) / 
        max_exact + jnp.finfo(jnp.float32).eps) / jnp.log(max_distance / 
        max_exact) * (num_buckets - max_exact)).astype(jnp.int32)
    val_y_if_large = jnp.minimum(val_y_if_large, num_buckets - 1)

    xx = jnp.where(is_small_x, nx, val_x_if_large)
    yy = jnp.where(is_small_y, ny, val_y_if_large)
    ret += xx + num_buckets * yy 
    return ret

  def rel_attention_text(self, txt_position_ids, bidirectional=True):
    txt_context_position = txt_position_ids[:, :, None]
    txt_memory_position = txt_position_ids[:, None, :]
    txt_relative_position = txt_memory_position - txt_context_position # shape (qlen, klen)

    # different way to compute relative position.
    rp_bucket = self._relative_position_bucket(
      txt_relative_position,
      bidirectional=bidirectional,
      num_buckets=self.num_buckets,
      max_distance=self.max_distance)

    relative_attention_bias = self.relative_attention_bias
    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    t_values = lax.dot_general(
      relative_attention_bias,
      rp_bucket_one_hot,
      (
        ((1,), (0,)),
        ((), ())))  # no batched dims
    return jnp.transpose(t_values, (1,0,2,3))

  def rel_attention_image(self, img_position_ids, bidirectional=True):
    img_position_x = img_position_ids % self.img_width
    img_position_y = img_position_ids // self.img_width
    img_context_position_x = img_position_x[:,:,None]
    img_memory_position_x = img_position_x[:, None, :]
    img_context_position_y = img_position_y[:,:,None]
    img_memory_position_y = img_position_y[:, None, :]
    img_relative_position_x = img_memory_position_x - img_context_position_x
    img_relative_position_y = img_memory_position_y - img_context_position_y

    img_rp_bucket = self._img_relative_position_bucket(
      img_relative_position_x,
      img_relative_position_y,
      num_buckets=self.img_num_buckets,
      max_distance=self.img_max_distance)

    image_num_rel_dis = self.img_num_buckets ** 2 * 4
    img_relative_attention_bias = self.img_relative_attention_bias
    img_relative_attention_bias = jnp.asarray(img_relative_attention_bias, self.dtype)

    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    img_bcast_iota = lax.broadcasted_iota(jnp.int32, (image_num_rel_dis, 1, 1, 1), 0)
    img_rp_bucket_one_hot = jnp.array(
      img_rp_bucket[jnp.newaxis, ...] == img_bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    i_values = lax.dot_general(
      img_relative_attention_bias,
      img_rp_bucket_one_hot,
      (
        ((1,), (0,)),  # rhs, lhs contracting dims
        ((), ())))  # no batched dims
    return jnp.transpose(i_values, (1,0,2,3))

  def __call__(self, txt_position_ids, img_position_ids, bidirectional=True):
    """Produce relative position embedding attention biases.

    Args:
      txt_position_ids: attention query length.
      img_position_ids: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.

    Returns:
      output: `(1, len, q_len, k_len)` attention bias
    """
    # TODO(levskaya): should we be computing this w. numpy as a program
    # constant?

    # compute text position encoding first.
    txt_context_position = txt_position_ids[:, :, None]
    txt_memory_position = txt_position_ids[:, None, :]
    txt_relative_position = txt_memory_position - txt_context_position # shape (qlen, klen)
    
    # different way to compute relative position.
    rp_bucket = self._relative_position_bucket(
        txt_relative_position,
        bidirectional=bidirectional,
        num_buckets=self.num_buckets,
        max_distance=self.max_distance)

    relative_attention_bias = self.relative_attention_bias

    img_position_x = img_position_ids % self.img_width
    img_position_y = img_position_ids // self.img_width
    img_context_position_x = img_position_x[:,:,None]
    img_memory_position_x = img_position_x[:, None, :]
    img_context_position_y = img_position_y[:,:,None]
    img_memory_position_y = img_position_y[:, None, :]
    img_relative_position_x = img_memory_position_x - img_context_position_x
    img_relative_position_y = img_memory_position_y - img_context_position_y

    img_rp_bucket = self._img_relative_position_bucket(
        img_relative_position_x,
        img_relative_position_y,
        num_buckets=self.img_num_buckets,
        max_distance=self.img_max_distance)

    image_num_rel_dis = self.img_num_buckets ** 2 * 4
    img_relative_attention_bias = self.img_relative_attention_bias

    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    img_relative_attention_bias = jnp.asarray(img_relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    
    img_bcast_iota = lax.broadcasted_iota(jnp.int32, (image_num_rel_dis, 1, 1, 1), 0)
    img_rp_bucket_one_hot = jnp.array(
        img_rp_bucket[jnp.newaxis, ...] == img_bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    t_values = lax.dot_general(
        relative_attention_bias, 
        rp_bucket_one_hot, 
        (
            ((1,), (0,)),
            ((), ())))  # no batched dims
    i_values = lax.dot_general(
        img_relative_attention_bias,
        img_rp_bucket_one_hot,
        (
            ((1,), (0,)),  # rhs, lhs contracting dims
            ((), ())))  # no batched dims

    t_values_pad = jax.lax.pad(
        t_values, 
        jnp.array(0, dtype=t_values.dtype), 
        [(0,0,0),(0,0,0),(0,img_position_ids.shape[1],0),(0,img_position_ids.shape[1],0)])
    
    i_values_pad = jax.lax.pad(
        i_values, 
        jnp.array(0, dtype=i_values.dtype), 
        [(0,0,0),(0,0,0),(txt_position_ids.shape[1],0,0),(txt_position_ids.shape[1],0,0)])
    values = t_values_pad + i_values_pad
    out = jnp.transpose(values, (1,0,2,3))
    return out


class LengthenMLP(nn.Module):
  n: int
  dtype: Any

  @compact
  def __call__(self, old_y, y, cur_index=None):
    layer = DenseGeneral(
      features=self.n*y.shape[-1],
      use_bias=True,
      kernel_axes=('embed', 'mlp'),
      bias_axes=('mlp',),
      dtype=self.dtype
    )
    batch, seq, dim = y.shape
    y = layer(y)
    if cur_index is None:
      y = jnp.reshape(y, [batch, seq*self.n, dim])
      return y + old_y
    else:
      # TODO could cache this value
      y = jnp.reshape(y, [batch, self.n, seq, dim])
      ix = cur_index % self.n
      return lax.dynamic_slice_in_dim(y, ix, 1, axis=1)[:, 0] + old_y


class ShortenMLP(nn.Module):
  n: int
  dtype: Any
  cfg: Any

  @compact
  def __call__(self, y, decoder_attn_mask, encoder_decoder_mask, decoder_bias, cross_abs_pos_bias,
               subsegments=None, cur_index=None, decode=False):
    """
    Args:
      y: [batch, seq_len, dim]
      decoder_attn_mask: [batch, n_heads, seq_len, seq_len]
      encoder_decoder_mask: [batch, n_heads, seq_len, input_seq_len]
      decoder_bias: [batch, n_heads, seq_len, seq_len]
    """
    batch, seq_len, dim = y.shape
    if cur_index is not None:
      seq_len = decoder_bias.shape[2]
    new_len = seq_len // self.n  # Output sequence length
    if seq_len % self.n != 0:
      raise ValueError(f"Cannot shorten seq len {seq_len} by {self.n}")
    shift = self.n - 1  # How much to shift the embeddings right
    input_mask = decoder_attn_mask

    # State to insert to the right when shifting
    shift_vals = param_with_axes(
      'kernel',
      nn.initializers.uniform(0.01, jnp.float32),
      (shift, y.shape[-1]),
      y.dtype,
      axes=('embed', 'joined_kv')
    )

    # MLP to merge consecutive states
    # logging.info("Using attention pooling")
    # layer = AttentionPooling(
    #   name="atten_pool",
    #   n_inputs=self.n,
    #   n_outputs=1,
    #   cfg=self.cfg
    # )
    layer = DenseGeneral(
      features=dim,
      use_bias=True,
      dtype=self.dtype,
      kernel_axes=('embed', 'mlp'),
      bias_axes=('mlp',),
    )

    # Bias is full length even when decoding, so we always compute the new full-sized bias matrix
    decoder_bias = jnp.pad(
      decoder_bias[:, :, :-shift, :-shift],
      [[0, 0], [0, 0], [shift, 0], [shift, 0]]
    )
    # Merge cells just by averaging
    decoder_bias = jnp.reshape(decoder_bias,
                               list(decoder_bias.shape[:2]) + [new_len, self.n, new_len, self.n])

    if decode and cur_index is None:
      # Initializing the decoding cache
      # The previous `self.n-1` tokens we are accumulating
      self.variable('cache', 'new_token', lambda: jnp.repeat(jnp.expand_dims(shift_vals, 0), batch, 0))
      # The previous token we return output
      self.variable('cache', 'last_token', jnp.zeros, (batch, dim), y.dtype)
      if cross_abs_pos_bias is not None:
        b, heads, _, encoder_len = cross_abs_pos_bias.shape
        self.variable('cache', 'prev_bias', jnp.zeros,
                      (b, heads*encoder_len), cross_abs_pos_bias.dtype)

    if cur_index is not None:
      assert subsegments is None
      assert decoder_attn_mask is None
      assert encoder_decoder_mask.shape[2] == 1   # Mask is should be the same each token

      decoder_bias = jnp.mean(decoder_bias, -3)
      decoder_bias = jnp.mean(decoder_bias, -1)

      def _save_to_cache(mdl, layer_, y_, _bias):
        # Write the incoming token to the cache and return the previous token
        new_tokens = mdl.variable('cache', 'new_token')
        last_token = mdl.variable('cache', 'last_token')
        if _bias is not None:
          prev_bias = mdl.variable('cache', 'prev_bias')
          prev_bias.value += jnp.reshape(_bias, prev_bias.value.shape)

        mask = ((cur_index - 1) % self.n) == jnp.arange(self.n-1)[None, :, None]
        new_tokens.value = new_tokens.value * jnp.logical_not(mask) + mask * y_
        return last_token.value, _bias

      def _decode(mdl, layer_, y_, _bias):
        # Use the cache and new token to compute the new super-token
        new_tokens = mdl.variable('cache', 'new_token')
        last_token = mdl.variable('cache', 'last_token')
        # [n, batch, dim]
        last_n_tokens = jnp.concatenate([new_tokens.value, y_], 1)  # [batch, n, dim]
        last_n_tokens = jnp.reshape(last_n_tokens, [batch, self.n*dim])
        token = jnp.squeeze(layer_(jnp.expand_dims(last_n_tokens, 1)), 1)
        last_token.value = token
        if _bias is not None:
          prev = mdl.variable('cache', 'prev_bias')
          _bias = (jnp.reshape(prev.value, _bias.shape) + _bias) / self.n
          prev.value *= 0
        return token, _bias

      y, cross_abs_pos_bias = linen.cond(
        cur_index % self.n == 0,
        _decode,
        _save_to_cache,
        self, layer, y, cross_abs_pos_bias
      )
      return jnp.expand_dims(y, 1), decoder_attn_mask, encoder_decoder_mask, decoder_bias, cross_abs_pos_bias
    else:
      decoder_attn_mask = jnp.pad(
        decoder_attn_mask[:, :, :-shift, :-shift],
        [[0, 0], [0, 0], [shift, 0], [shift, 0]]
      )
      decoder_attn_mask = jnp.reshape(
        decoder_attn_mask, list(decoder_attn_mask.shape[:2]) + [new_len, self.n, new_len, self.n])

      enc_len = encoder_decoder_mask.shape[-1]
      encoder_decoder_mask = jnp.pad(
        encoder_decoder_mask[:, :, :-shift],
        [[0, 0], [0, 0], [shift, 0], [0, 0]]
      )
      encoder_decoder_mask = jnp.reshape(
        encoder_decoder_mask, list(encoder_decoder_mask.shape[:2]) + [new_len, self.n, enc_len])

      if cross_abs_pos_bias is not None:
        cross_abs_pos_bias = jnp.pad(
          cross_abs_pos_bias[:, :, :-shift],
          [[0, 0], [0, 0], [shift, 0], [0, 0]]
        )
        cross_abs_pos_bias = jnp.reshape(
          cross_abs_pos_bias, list(cross_abs_pos_bias.shape[:2]) + [new_len, self.n, enc_len])

      # Shift y by self.n-1
      if subsegments is not None:
        # if there are subsegments we have to be careful not to shift a token that ends one
        # sub-segment into the start of a new subsegment, instead each individual sub-segment
        # should start with `shift_vals` and the initial token to match decoding it individually

        # Shift y right by `shift` and pad with zeros for now`
        y = jnp.reshape(y, [batch, new_len*self.n, dim])
        y = jnp.pad(y[:, :-shift], [[0, 0], [shift, 0], [0, 0]])
        y = jnp.reshape(y, [batch, new_len, self.n, dim])

        # Subsegment of the super-tokens, note we assume subsegments have already been padded to be
        # a multiple of `self.n` shape=[batch, new_len]
        shortened_subsegments = lax.slice_in_dim(subsegments, 0, seq_len, stride=self.n, axis=1)

        # Which super-tokens represent a change in subs
        subsegment_shifts = jnp.pad(shortened_subsegments[:, 1:] != shortened_subsegments[:, :-1],
                                    [[0, 0], [1, 0]], constant_values=True)
        # [batch, new_len, self.n]
        # False for tokens that will be replace with `shifts_vals`, that is, tokens
        # that will be replaced with rotate-in blank tokens
        subsegment_mask = jnp.logical_or(jnp.logical_not(subsegment_shifts)[:, :, None],
                                         jnp.array([False]*(self.n-1) + [True])[None, None, :])

        # Zero out embeddings/biases/masks for tokens that will be replaced with shift_val
        y = y * subsegment_mask[:, :, :, None]
        decoder_bias = decoder_bias * subsegment_mask[:, None, None, None, :, :]
        decoder_bias = decoder_bias * subsegment_mask[:, None, :, :, None, None]
        encoder_decoder_mask = encoder_decoder_mask * subsegment_mask[:, None, :, :, None]
        if cross_abs_pos_bias is not None:
          cross_abs_pos_bias = cross_abs_pos_bias * subsegment_mask[:, None, :, :, None]
        decoder_attn_mask = decoder_attn_mask * subsegment_mask[:, None, :, :, None, None]
        decoder_attn_mask = decoder_attn_mask * subsegment_mask[:, None, None, None, :, :]

        # Now replace those zeroed-out tokens with `shift_vals`
        proj = jnp.reshape(jnp.pad(shift_vals, [[0, 1], [0, 0]]), (1, 1, self.n, dim))
        y = y + (proj*subsegment_shifts[:, :, None, None])
      else:
        # No subsegments means this is a simple shift
        y = jnp.reshape(y, [batch, new_len*self.n, dim])
        shift_vals = jnp.repeat(jnp.expand_dims(shift_vals, 0), batch, 0)
        y = jnp.concatenate([shift_vals, y[:, :-shift]], axis=1)

      # Now do the shortening
      decoder_bias = jnp.mean(decoder_bias, -3)
      decoder_bias = jnp.mean(decoder_bias, -1)
      encoder_decoder_mask = jnp.sum(encoder_decoder_mask, -2) > 0
      if cross_abs_pos_bias is not None:
        cross_abs_pos_bias = jnp.mean(cross_abs_pos_bias, -2)
      decoder_attn_mask = jnp.sum(decoder_attn_mask, -3)
      decoder_attn_mask = jnp.sum(decoder_attn_mask, -1)
      decoder_attn_mask = decoder_attn_mask > 0

      y = jnp.reshape(y, [batch, new_len, self.n*dim])
      y = layer(y)
      return y, decoder_attn_mask, encoder_decoder_mask,  decoder_bias, cross_abs_pos_bias


class TextRelativeAttention(nn.Module):
  num_heads: int
  dtype: Any
  num_buckets: int=32
  max_distance: int=128
  embedding_init: Callable = nn.linear.default_embed_init

  def setup(self) -> None:
    self.relative_attention_bias = param_with_axes(
      'rel_embedding',
      self.embedding_init, (self.num_heads, self.num_buckets),
      jnp.float32,
      axes=('heads', 'relpos_buckets'))

  @staticmethod
  def _relative_position_bucket(
      relative_position, bidirectional=False, num_buckets=32, max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(jnp.int32) * num_buckets
      n = jnp.abs(n)
    else:
      n = jnp.maximum(n, 0)

    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        jnp.log(n.astype(jnp.float32) / max_exact + jnp.finfo(jnp.float32).eps) /
        jnp.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(jnp.int32)

    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    ret += jnp.where(is_small, n, val_if_large)
    return ret

  def __call__(self, txt_position_ids, bidirectional=True):
    txt_context_position = txt_position_ids[:, :, None]
    txt_memory_position = txt_position_ids[:, None, :]
    txt_relative_position = txt_memory_position - txt_context_position # shape (qlen, klen)

    # different way to compute relative position.
    rp_bucket = self._relative_position_bucket(
      txt_relative_position,
      bidirectional=bidirectional,
      num_buckets=self.num_buckets,
      max_distance=self.max_distance)

    relative_attention_bias = self.relative_attention_bias
    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    t_values = lax.dot_general(
      relative_attention_bias,
      rp_bucket_one_hot,
      (
        ((1,), (0,)),
        ((), ())))  # no batched dims
    return jnp.transpose(t_values, (1,0,2,3))


class ImageRelativeAttention(nn.Module):
  num_heads: int
  img_width: int
  img_height: int
  dtype: Any
  embedding_init: Callable = nn.linear.default_embed_init
  img_max_distance: int = 20
  img_num_buckets: int = 8

  @staticmethod
  def _img_relative_position_bucket(relative_position_x,
                                    relative_position_y,
                                    num_buckets=8,
                                    max_distance=20):

    max_exact = num_buckets // 2
    nx = -relative_position_x
    ny = -relative_position_y

    total_buckets = num_buckets ** 2
    ret = 0
    ret += (jnp.logical_and(nx <=0, ny <0)).astype(jnp.int32) * total_buckets * 3
    ret += (jnp.logical_and(nx <0, ny >=0)).astype(jnp.int32) * total_buckets * 2
    ret += (jnp.logical_and(nx >0, ny <=0)).astype(jnp.int32) * total_buckets * 1

    nx = jnp.abs(nx)
    ny = jnp.abs(ny)

    is_small_x = nx < max_exact
    val_x_if_large = max_exact + (jnp.log(nx.astype(jnp.float32) /
                                          max_exact + jnp.finfo(jnp.float32).eps) / jnp.log(max_distance /
                                                                                            max_exact) * (num_buckets - max_exact)).astype(np.int32)

    val_x_if_large = jnp.minimum(val_x_if_large, num_buckets - 1)

    is_small_y = ny < max_exact
    val_y_if_large = max_exact + (jnp.log(ny.astype(jnp.float32) /
                                          max_exact + jnp.finfo(jnp.float32).eps) / jnp.log(max_distance /
                                                                                            max_exact) * (num_buckets - max_exact)).astype(jnp.int32)
    val_y_if_large = jnp.minimum(val_y_if_large, num_buckets - 1)

    xx = jnp.where(is_small_x, nx, val_x_if_large)
    yy = jnp.where(is_small_y, ny, val_y_if_large)
    ret += xx + num_buckets * yy
    return ret

  def setup(self) -> None:
    image_num_rel_dis = self.img_num_buckets ** 2 * 4
    self.relative_attention_bias = param_with_axes(
      'rel_embedding',
      self.embedding_init, (self.num_heads, image_num_rel_dis),
      jnp.float32,
      axes=('heads', 'relpos_buckets'))

  def __call__(self, img_position_ids):
    img_position_x = img_position_ids % self.img_width
    img_position_y = img_position_ids // self.img_width
    img_context_position_x = img_position_x[:,:,None]
    img_memory_position_x = img_position_x[:, None, :]
    img_context_position_y = img_position_y[:,:,None]
    img_memory_position_y = img_position_y[:, None, :]
    img_relative_position_x = img_memory_position_x - img_context_position_x
    img_relative_position_y = img_memory_position_y - img_context_position_y

    img_rp_bucket = self._img_relative_position_bucket(
      img_relative_position_x,
      img_relative_position_y,
      num_buckets=self.img_num_buckets,
      max_distance=self.img_max_distance)

    image_num_rel_dis = self.img_num_buckets ** 2 * 4
    img_relative_attention_bias = self.relative_attention_bias
    img_relative_attention_bias = jnp.asarray(img_relative_attention_bias, self.dtype)

    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    img_bcast_iota = lax.broadcasted_iota(jnp.int32, (image_num_rel_dis, 1, 1, 1), 0)
    img_rp_bucket_one_hot = jnp.array(
      img_rp_bucket[jnp.newaxis, ...] == img_bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    i_values = lax.dot_general(
      img_relative_attention_bias,
      img_rp_bucket_one_hot,
      (
        ((1,), (0,)),  # rhs, lhs contracting dims
        ((), ())))  # no batched dims

    return jnp.transpose(i_values, (1,0,2,3))


#------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
#------------------------------------------------------------------------------
class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Initializer = nn.initializers.ones
  use_scale: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    
    if self.use_scale:
      scale = param_with_axes(
        'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))
      scale = jnp.asarray(scale, self.dtype)
      y = y * scale
    
    return y

def _canonicalize_axes(rank: int, axes: Axes) -> Iterable[int]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))

def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)

def _compute_stats(x: Array, axes: Axes,
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None):
  """Computes mean and variance statistics.
  This implementation takes care of a few important details:
  - Computes in float32 precision for half precision inputs
  -  mean and variance is computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.
  """
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(_abs_sq(x), axes)
  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - _abs_sq(mean))
  return mean, var

def _normalize(mdl: Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Any, param_dtype: Any,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Initializer,
               scale_init: Initializer):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.
  A seperate bias and scale is learned for each feature as specified by feature_axes.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if use_scale:
    scale = param_with_axes('scale', scale_init, reduced_feature_shape,
                      param_dtype,  axes=('axis_0',)).reshape(feature_shape)
    mul *= scale
  y *= mul
  if use_bias:
    bias = param_with_axes('bias', bias_init, reduced_feature_shape,
                     param_dtype,  axes=('axis_0',)).reshape(feature_shape)
    y += bias
  return jnp.asarray(y, dtype)

class LayerNormWithBias(nn.Module):
  """
  Layer normalization for the Vision Transformer.
  """
  epsilon: float = 1e-6
  dtype: Optional[Any] = None
  param_dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = initializers.zeros
  scale_init: Initializer = initializers.ones
  reduction_axes: Axes = -1
  feature_axes: Axes = -1
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

  @compact
  def __call__(self, x):
    """Applies layer normalization on the input.
    Args:
      x: the inputs
    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(x, self.reduction_axes,
                               self.axis_name, self.axis_index_groups)

    return _normalize(
        self, x, mean, var, self.reduction_axes, self.feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)

def _canonicalize_axes(rank: int, axes: Axes) -> Iterable[int]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))

def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)

def _compute_stats(x: Array, axes: Axes,
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None):
  """Computes mean and variance statistics.
  This implementation takes care of a few important details:
  - Computes in float32 precision for half precision inputs
  -  mean and variance is computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.
  """
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(_abs_sq(x), axes)
  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - _abs_sq(mean))
  return mean, var

def _normalize(mdl: Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Any, param_dtype: Any,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Initializer,
               scale_init: Initializer):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.
  A seperate bias and scale is learned for each feature as specified by feature_axes.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if use_scale:
    scale = param_with_axes('scale', scale_init, reduced_feature_shape,
                      param_dtype,  axes=('axis_0',)).reshape(feature_shape)
    mul *= scale
  y *= mul
  if use_bias:
    bias = param_with_axes('bias', bias_init, reduced_feature_shape,
                     param_dtype,  axes=('axis_0',)).reshape(feature_shape)
    y += bias
  return jnp.asarray(y, dtype)
  


class GroupNorm(Module):
  num_groups: Optional[int] = 32
  group_size: Optional[int] = None
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  param_dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = nn.initializers.zeros
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    reduction_axes = list(range(1, x.ndim - 1)) + [-1]
    feature_axes = (-1,)

    if ((self.num_groups is None and self.group_size is None) or
        (self.num_groups is not None and self.group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be '
                       'specified, but not both of them.')
    num_groups = self.num_groups

    channels = x.shape[-1]
    if self.group_size is not None:
      if channels % self.group_size != 0:
        raise ValueError('Number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(channels, self.group_size))
      num_groups = channels // self.group_size

    if num_groups <= 0 or channels % num_groups != 0:
      raise ValueError('Number of groups ({}) does not divide the number'
                       ' of channels ({}).'.format(num_groups, channels))

    group_size = x.shape[-1] // num_groups
    group_shape = x.shape[:-1] + (num_groups, group_size)

    def broadcast_stat(stat):
      stat = jnp.broadcast_to(stat[..., None], (x.shape[0], num_groups, group_size))
      return stat.reshape((x.shape[0], num_groups * group_size))

    # TODO suport axis_name for model parallelism?
    mean, var = _compute_stats(x.reshape(group_shape), reduction_axes, None, None)
    mean = broadcast_stat(mean)
    var = broadcast_stat(var)

    return _normalize(
        self, x, mean, var, reduction_axes[:-1], feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)

class SpatialNorm(Module):
  in_channels: int 
  add_conv: bool = False
  params_init: Any = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, f: jnp.ndarray, zq: jnp.ndarray) -> jnp.ndarray:

    zq = jax.image.resize(zq, shape=f.shape[:3] + (zq.shape[-1],), method='nearest')

    scale_init = one_init if self.params_init is None else lambda *_ : jnp.array(self.params_init['norm_layer']['weight'])
    bias_init = zero_init if self.params_init is None else lambda *_ : jnp.array(self.params_init['norm_layer']['bias']) 
    norm_f = GroupNorm(
        use_bias = True,
        use_scale = True,
        bias_init = bias_init,
        scale_init = scale_init,
        name='norm_layer')(f)

    w_init = default_init if self.params_init is None else lambda *_ : jnp.transpose(jnp.array(self.params_init['conv_y']['weight']), (2, 3, 1, 0))
    b_init = zero_init if self.params_init is None else lambda *_ : jnp.array(self.params_init['conv_y']['bias']) 

    conv_y = Conv(
      features=self.in_channels,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='convy')(zq)

    w_init = default_init if self.params_init is None else lambda *_ : jnp.transpose(jnp.array(self.params_init['conv_b']['weight']), (2, 3, 1, 0))
    b_init = zero_init if self.params_init is None else lambda *_ : jnp.array(self.params_init['conv_b']['bias']) 

    conv_b = Conv(
      features=self.in_channels,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='convb')(zq)

    return norm_f * conv_y + conv_b
#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: DType = jnp.float32) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
      # [batch, len_q] -> [batch, len_q, 1]
      jnp.expand_dims(query_input, axis=-1),
      # [batch, len_q] -> [batch, 1, len_kv]
      jnp.expand_dims(key_input, axis=-2))

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)

def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
  will be `[batch, heads, len, len]` and this function will produce a
  causal mask of shape `[batch, 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch, len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: DType,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None) -> Array:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]
    decoder_segment_ids: decoder segmentation info for packed examples. [batch,
      length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attneition` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, lengtlength]
    inputs_mask = make_attention_mask(
        decoder_causal_attention,
        decoder_causal_attention,
        jnp.logical_and,
        dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      make_attention_mask(
          decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
        make_attention_mask(
            decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

  return combine_masks(*masks, dtype=dtype)


class AdditivePositionEmbs(nn.Module):
  """Additive learned positional embeddings to the inputs.
  Attributes:
    posemb_init: positional embedding initializer.
  """
  posemb_init: Initializer
  dtype: jnp.float32

  @nn.compact
  def __call__(self, input_shape):
    """Applies the AdditivePositionEmbs module.
    Args:
      inputs: Inputs to the layer.
    Returns:
      Output tensor with shape `(timesteps, in_dim)`.
    """
    # inputs.shape is (seq_len, emb_dim).
    # assert inputs.ndim == 3, ('Number of dimensions should be 3,'
    #                           ' but it is: %d' % inputs.ndim)
    pe = param_with_axes(
            'pos_embedding',
            self.posemb_init, input_shape,
            jnp.float32,
            axes=(('image_patch', 'embed')))

    pe = jnp.array(pe, self.dtype)
    return pe


class Additive2DPositionEmbs(nn.Module):
  """Additive learned positional embeddings to the inputs.
  Attributes:
    posemb_init: positional embedding initializer.
  """
  posemb_init: Initializer
  dtype: jnp.float32

  @nn.compact
  def __call__(self, input_shape):
    """Applies the AdditivePositionEmbs module.
    Args:
      inputs: Inputs to the layer.
    Returns:
      Output tensor with shape `(timesteps, in_dim)`.
    """
    # inputs.shape is (seq_len, emb_dim).
    # assert inputs.ndim == 3, ('Number of dimensions should be 3,'
    #                           ' but it is: %d' % inputs.ndim)
    row_embedding = param_with_axes(
            'row_embedding',
            self.posemb_init, (input_shape[0], input_shape[-1]),
            jnp.float32,
            axes=(('image_patch', 'embed')))

    col_embedding = param_with_axes(
            'col_embedding',
            self.posemb_init, (input_shape[1], input_shape[-1]),
            jnp.float32,
            axes=(('image_patch', 'embed'))) 

    row_ids = jnp.arange(input_shape[0])
    col_ids = jnp.arange(input_shape[1])
    pos_emb = row_embedding[row_ids][:,None,:] + col_embedding[col_ids][None,:,:]
    pos_emb = jnp.reshape(pos_emb, [-1, pos_emb.shape[-1]])

    pos_emb = jnp.array(pos_emb, self.dtype)
    return pos_emb

def get_sinusoid_encoding_table(seq_length, emb_dim, dtype):
  """Sinusoid position encoding table: excerpt from original Transformer"""
  def get_position_angle_vec(position):
    return [
      position / np.power(10000, 2 * (dim_j // 2) / emb_dim)
      for dim_j in range(emb_dim)
    ]
  
  sinusoid_table = np.array(
    [get_position_angle_vec(pos_i) for pos_i in range(seq_length)]
  )
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
  
  pos_emb = jnp.array(sinusoid_table).astype(dtype)
  return pos_emb


# Layers for ViT-VQGAN implementation.
def get_2d_sincos_pos_embed(emb_dim, image_size, image_patch_size, dtype, class_token=False, temperature=10000.):
  """
  (Absolute, additive) 2D sinusoidal positional embeddings used in MoCo v3, MAE
  Args:
    emb_dim (int): embedding dimension
    image_size (tuple): image size
    image_patch_size (int): image patch size
    class_token (bool): whether to use class token
  """
  h, w = image_size[0] // image_patch_size[0], image_size[1] // image_patch_size[1]
  grid_h = jnp.arange(h, dtype=jnp.float32)
  grid_w = jnp.arange(w, dtype=jnp.float32)
  grid_w, grid_h = jnp.meshgrid(grid_w, grid_h, indexing='xy')

  assert emb_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
  emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_w, jnp.float32, temperature) # (H*W, D/2)
  emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_h, jnp.float32, temperature) # (H*W, D/2)
  pos_emb = jnp.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
  if class_token:
    pos_emb = jnp.concatenate([jnp.zeros([1, emb_dim], dtype=pos_emb.dtype), pos_emb], axis=0)
  pos_emb = pos_emb.astype(dtype)
  return pos_emb


def get_1d_sincos_pos_embed_from_grid(emb_dim, pos, dtype, temperature=10000.):
  """
  (Absolute, additive) 1D sinusoidal positional embeddings used in MoCo v3, MAE
  Args:
    emb_dim (int):output dimension for each position
    pos: a list of positions to be encoded: size (M, )
    out: (M, D)
  """
  assert emb_dim % 2 == 0
  omega = jnp.arange(emb_dim // 2, dtype=jnp.float32)
  omega /= emb_dim / 2.
  omega = 1. / temperature**omega  # (D/2,)

  pos = pos.reshape(-1).astype(jnp.float32)  # (M,)
  out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

  emb_sin = jnp.sin(out) # (M, D/2)
  emb_cos = jnp.cos(out) # (M, D/2)

  emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
  return emb.astype(dtype)


def get_rotary_coordinates(seq_len, dtype=jnp.float32, center_origin=True, llama=False):
    """
    Get rotary coordinates for a single dimension
    :param seq_len: length of sequence (or dimension)
    :param dtype: data type
    :param center_origin: If true then coordinates are from [-seq_len / 2, seq_len / 2].
                          if false then coordinates from    [1, seq_len]
    :return: sequence of length L -- coordinates are from [-L / 2, -L / 2] if center_origin else [1, L]
    """

    assert dtype == jnp.float32
    if center_origin:
        sl0 = seq_len // 2
        nseq = jnp.arange(sl0, dtype=dtype) - float(sl0)
        pseq = 1.0 + jnp.arange(seq_len - sl0, dtype=dtype)
        return jnp.concatenate([nseq, pseq], 0)

    offset = 0.0 if llama else 1.0
    return offset + jnp.arange(seq_len, dtype=dtype)


def get_rotary_coordinates_2d(h, w, dtype=jnp.float32, llama=False, resolution=1):
    """
    Rotary embeddings for 2d (e.g. an image).
    Scale kinda like we're in a square box and taking a crop. skip zero though
    :param h: How many patches width
    :param w: How many patches height
    :param dtype: dtype
    :return: [h * w, 2] array of coords
    """

    assert dtype == jnp.float32
    base_scale = 1 if llama else 1 / (max(h, w) + 1.0)
    base_scale *= resolution
    w_coords = base_scale * get_rotary_coordinates(w, dtype=dtype, center_origin=False if llama else True, llama=llama)
    h_coords = base_scale * get_rotary_coordinates(h, dtype=dtype, center_origin=False if llama else True, llama=llama)
    return jnp.stack(jnp.meshgrid(h_coords, w_coords, indexing='ij'), -1).reshape((h * w, 2))


def multimodal_rotary_coords(h=None, w=None, token_idx=None, modality_idx=None, dtype=jnp.float32,
                             max_token=1024, max_modality=5):
    """
    Rotary embeddings for the multimodal transformer
    :param h: [L] h coords (default to 0.0 otherwise)
    :param w: [L] w coords (default to 0.0 otherwise)
    :param token_idx: [L] token_idx coords (default to 0.0 otherwise)
    :param modality_idx: [L] modality_idx coords (default to 0.0 otherwise)
    :param dtype: final datatype
    :return: [L, 4] rotary coords
    """
    assert dtype == jnp.float32

    ls = [x.shape[0] for x in [h, w, token_idx, modality_idx] if x is not None]
    L = ls[0]
    assert all([x == L for x in ls])

    h_vec = jnp.zeros([L], dtype=dtype) if (h is None) else h
    w_vec = jnp.zeros([L], dtype=dtype) if (w is None) else w
    t_vec = jnp.zeros([L], dtype=dtype) if (token_idx is None) else token_idx / max_token
    m_vec = jnp.zeros([L], dtype=dtype) if (modality_idx is None) else modality_idx / max_modality
    return jnp.stack([h_vec, w_vec, t_vec, m_vec], -1)


def llama_multimodal_rotary_coords(h=None, w=None, token_idx=None, latent_idx=None, dtype=jnp.float32):
    """
    Rotary embeddings for the multimodal transformer
    :param h: [L] h coords (default to 0.0 otherwise)
    :param w: [L] w coords (default to 0.0 otherwise)
    :param token_idx: [L] token_idx coords (default to 0.0 otherwise)
    :param latent_idx: [L] latent_idx coords (default to 0.0 otherwise)
    :param dtype: final datatype
    :return: [L, 4] rotary coords
    """
    ls = [x.shape[0] for x in [h, w, token_idx, latent_idx] if x is not None]
    L = ls[0]
    assert all([x == L for x in ls])
    assert dtype == jnp.float32

    h_vec = jnp.zeros([L], dtype=dtype) if (h is None) else h
    w_vec = jnp.zeros([L], dtype=dtype) if (w is None) else w
    t_vec = jnp.zeros([L], dtype=dtype) if (token_idx is None) else token_idx
    l_vec = jnp.zeros([L], dtype=dtype) if (latent_idx is None) else latent_idx
    return jnp.stack([h_vec, w_vec, t_vec, l_vec], -1)


def construct_rotary_sinusoids(coords, rotary_hsize: int = 32, llama=False, max_freq=10.0, dtype=jnp.float32):
    """
    :param coords: [*batch_dims, seq_length, num_dimensions]
    :param rotary_hsize: How many dimensions we will finally use during the rotary embs
    :param max_freq: We will have frequencies that take the entire sequence (in the range of [0, 1]) as the first
                     one, up until take 1/max_freq of the entire sequence, in a logarithmic sequence
    :return: Sinusoids of size [*batch_dims, seq_len, 2 (cos then sin) * rotary_hsize]
             they are repeated accordingly
    """
    assert dtype == jnp.float32
    # Sanity check
    *batch_dims, seq_length, num_dims = coords.shape
    if not llama:
      rotary_hsize = rotary_hsize // 2
    assert rotary_hsize % (num_dims * 2) == 0
    dim_expansion = rotary_hsize // (num_dims * 2)
    assert dim_expansion > 0

    if llama:
      dim = dim_expansion * 2
      freqs = 1.0 / (
        10000.0 ** (jnp.arange(0, dim, 2)[:dim_expansion].astype(coords.dtype if dtype is None else dtype) / dim)
      )
    else:
      freqs = jnp.logspace(0.0, math.log2(max_freq / 2.0), dim_expansion, base=2,
                           dtype=coords.dtype if dtype is None else dtype)
      freqs = freqs * np.pi
    for i in range(len(batch_dims) + 2):
        freqs = freqs[None]

    radians = coords[..., None] * freqs
    radians = radians.reshape(*batch_dims, seq_length, num_dims * dim_expansion)
    cos_t = jnp.cos(radians)
    sin_t = jnp.sin(radians)

    # Here we're repeating on the final dimension
    # bc later we will go through the first rotary_hsize coordinates and do
    # sin'd part: [-x1, x0, -x3, x2, ....]
    # cos'd part: [x0,  x1,  x2, x3, ....]
    cos_t = jnp.repeat(cos_t, 2, axis=-1)
    sin_t = jnp.repeat(sin_t, 2, axis=-1)
    sinusoids = with_sharding_constraint(jnp.concatenate([cos_t, sin_t], -1), ('length', 'kv'))
    return sinusoids


# def apply_rotary(query_key, sinusoids):
#     """
#     note: there's possibly a bug here (it differs from the usual rotary embedding. but somehow we got good results
#           anyways. weird!)
#     :param query_key: The query, key, or both. [*batch_dims, seq_len, num_heads, size_per_head]
#     :param sinusoids:                      [*sin_batch_dims, seq_len, 2 * rotary_hsize <= size_per_head]
#     :return: query_key with rotary applied
#     """
#     *sin_batch_dims, seq_len, sinusoids_hsize = sinusoids.shape
#     *batch_dims, seq_len, num_heads, size_per_head = query_key.shape

#     assert sinusoids_hsize % 2 == 0
#     rotary_hsize = sinusoids_hsize // 2
#     assert rotary_hsize <= size_per_head
#     for i in range(len(batch_dims) - len(sin_batch_dims)):
#         sinusoids = sinusoids[None]
    
#     cos = sinusoids[..., None, :rotary_hsize]
#     sin = sinusoids[..., None, rotary_hsize:]

#     qk_rope = query_key[..., :rotary_hsize]

#     qk_rotated_two = jnp.stack(
#       [
#         -jax.lax.slice(qk_rope, [0] * (len(qk_rope.shape) - 1) + [1], list(qk_rope.shape), [1] * (len(qk_rope.shape) -1) + [2]),
#         jax.lax.slice(qk_rope, [0] * len(qk_rope.shape), list(qk_rope.shape), [1] * (len(qk_rope.shape) -1) + [2]),
#       ],
#       -1
#     ).reshape(qk_rope.shape)

#     qk_rope = qk_rope * cos + qk_rotated_two * sin
#     query_key = jnp.concatenate([qk_rope, query_key[..., rotary_hsize:]], -1)
#     return query_key


def apply_rotary(x, rope_cache):
    """
    """
    xshaped = jnp.reshape(jnp.array(x, jnp.float32), (*x.shape[:-1], -1, 2))
    rope_cache = jnp.reshape(rope_cache, (xshaped.shape[0], xshaped.shape[1], 1, xshaped.shape[3], 2))

    x_out2 = jnp.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = jnp.reshape(x_out2, x_out2.shape[:3] + (-1, ))
    return jnp.array(x_out2, x.dtype)

def build_llama_rope_cache_1d(seq_len: int, n_elem: int, base: float=10000.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (jnp.arange(0, n_elem, 2).astype(jnp.float32) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = jnp.arange(seq_len).astype(jnp.float32)

    # Calculate the product of position index and $\theta_i$
    idx_theta = jnp.outer(seq_idx, theta).astype(jnp.float32)  # type: ignore
    cache = jnp.stack([jnp.cos(idx_theta), jnp.sin(idx_theta)], axis=-1)
    cache = jnp.reshape(cache, [cache.shape[0], -1])
    return jnp.asarray(cache, dtype=dtype)

def build_llama_rope_cache_2d(shape: tuple, n_elem: int, base: float=10000.0, resolution: float=1.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    
    img_coords = get_rotary_coordinates_2d(shape[0], shape[1], llama=True, resolution=resolution)
    n_elem = n_elem // 2
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (jnp.arange(0, n_elem, 2).astype(jnp.float32) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    # seq_idx = np.arange(seq_len).astype(jnp.float32)

    # Calculate the product of position index and $\theta_i$
    idx_theta_0 = jnp.outer(img_coords[:,0], theta).astype('float32')  # type: ignore
    idx_theta_1 = jnp.outer(img_coords[:,1], theta).astype('float32')  # type: ignore

    idx_theta = jnp.concatenate([idx_theta_0, idx_theta_1], axis=-1)
    cache = jnp.stack([jnp.cos(idx_theta), jnp.sin(idx_theta)], axis=-1)
    cache = jnp.reshape(cache, [cache.shape[0], -1])
    return jnp.asarray(cache, dtype=dtype)

def get_1d_position_embedding(pos_emb_type, length, emb_dim, head_dim, is_token, modality_idx, dtype, prefix=''):
  if pos_emb_type == 'learnable':
    posemb_init = nn.initializers.normal(stddev=0.02)
    shape = (length, emb_dim)
    positional_embedding = AdditivePositionEmbs(posemb_init=posemb_init, dtype=dtype, name=prefix+'positional_embedding')(shape)
  elif pos_emb_type == 'sinusoidal':
    positional_embedding = get_sinusoid_encoding_table(length, emb_dim, dtype)
  elif pos_emb_type == "rope":
    token_idx = get_rotary_coordinates(length, center_origin=False, llama=False)
    modality_idx = jnp.full(token_idx.shape, modality_idx)
    positional_embedding = jnp.asarray(construct_rotary_sinusoids(
      multimodal_rotary_coords(token_idx=token_idx, modality_idx=modality_idx),
      rotary_hsize=head_dim,
      llama=False,
    ), dtype=dtype)
  elif pos_emb_type == "llama_rope":
    positional_embedding = build_llama_rope_cache_1d(length, head_dim, dtype=dtype)
  else:
    raise NotImplementedError(f"{pos_emb_type}: not supported")
  return positional_embedding


def get_2d_position_embedding(
  pos_emb_type, input_size, patch_size,
  emb_dim, head_dim, modality_idx, dtype, resolution=1, prefix='',
):
  if isinstance(patch_size, int):
    patch_size = (patch_size, patch_size)
  if pos_emb_type == 'learnable':
    posemb_init = nn.initializers.normal(stddev=0.02)
    length = int(input_size[0] / patch_size[0]) * int(input_size[1] / patch_size[1])
    shape = (length, emb_dim)
    # shape = (input_size[0] // patch_size[0], input_size[1] // patch_size[1], emb_dim)
    positional_embedding = AdditivePositionEmbs(posemb_init=posemb_init, dtype=dtype, name=prefix+'positional_embedding')(shape)
  elif pos_emb_type == '2d-learnable':
    posemb_init = nn.initializers.normal(stddev=0.02)
    shape = (input_size[0] // patch_size[0], input_size[1] // patch_size[1], emb_dim)
    positional_embedding = Additive2DPositionEmbs(posemb_init=posemb_init, dtype=dtype, name=prefix+'positional_embedding')(shape)
  elif pos_emb_type == 'sinusoidal':
    length = int(input_size[0] / patch_size[0]) * int(input_size[1] / patch_size[1])
    positional_embedding = get_sinusoid_encoding_table(length, emb_dim, dtype)
  elif pos_emb_type == '2d-sincos':
    positional_embedding = get_2d_sincos_pos_embed(
      emb_dim=emb_dim,
      image_size=input_size,
      image_patch_size=patch_size,
      dtype=dtype,
      class_token=False, 
    )
  elif pos_emb_type == "rope":
    shape = (input_size[0] // patch_size[0], input_size[1] // patch_size[1], emb_dim)
    img_coords = get_rotary_coordinates_2d(shape[0], shape[1], llama=False)
    h_coords = img_coords[..., 0]
    w_coords = img_coords[..., 1]
    modality_idx = jnp.full(h_coords.shape, modality_idx)
    positional_embedding = jnp.asarray(construct_rotary_sinusoids(
      multimodal_rotary_coords(h=h_coords, w=w_coords, modality_idx=modality_idx),
      rotary_hsize=head_dim,
      llama=False,
    ), dtype=dtype)
  elif pos_emb_type == "llama_rope":
    shape = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
    positional_embedding = build_llama_rope_cache_2d(shape, head_dim, resolution=resolution, dtype=dtype)
 
  else:
    raise NotImplementedError(f"{pos_emb_type}: not supported")
  return positional_embedding


def drop_path(x: jnp.array, rng, drop_rate: float = 0.) -> jnp.array:
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
  This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
  the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
  See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
  changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
  'survival rate' as the argument.
  """
  if drop_rate == 0.:
      return x
  keep_prob = 1. - drop_rate
  mask = random.bernoulli(key=rng, p=keep_prob, shape=(x.shape[0],) + (1,)*(x.ndim-1))
  mask = jnp.broadcast_to(mask, x.shape)
  return lax.select(mask, (x / keep_prob).astype(x.dtype), jnp.zeros_like(x))


class DropPath(nn.Module):
  rate: float = 0.
  deterministic: Optional[bool] = None
  
  @nn.compact
  def __call__(self, x, deterministic: bool):
    deterministic = merge_param(
        'deterministic', self.deterministic, deterministic)
    if deterministic or self.rate == 0.:
        return x
    else:
      rng = self.make_rng('drop_path')
    return drop_path(x, rng, self.rate)


def canonicalize_padding(padding: Any, rank: int) -> Any:
  """"Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    f' int or pair of ints.')


class ConvTranspose(Module):
  """Convolution Module wrapping lax.conv_transpose.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: a sequence of `n` integers, representing the inter-window strides.
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """
  features: int
  kernel_size: Union[int, Tuple[int, ...]]
  strides: Optional[Tuple[int, ...]] = None
  padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
  kernel_dilation: Optional[Sequence[int]] = None
  use_bias: bool = True
  mask: Optional[Array] = None
  dtype: Any = None
  param_dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  kernel_axes: Tuple[str, ...] = ()
  bias_axes: Tuple[str, ...] = ()

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a transposed convolution to the inputs.

    Behaviour mirrors of `jax.lax.conv_transpose`.

    Args:
      inputs: input data with dimensions (*batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note: this is different from
        the input convention used by `lax.conv_general_dilated`, which puts the
        spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """
    kernel_size: Tuple[int, ...]
    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (
          (total_batch_size,) + inputs.shape[num_batch_dimensions:])
      inputs = jnp.reshape(inputs, flat_input_shape)

    strides: Tuple[int, ...]
    strides = self.strides or (1,) * (inputs.ndim - 2)

    in_features = jnp.shape(inputs)[-1]
    kernel_shape = kernel_size + (in_features, self.features)

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError('Mask needs to have the same shape as weights. '
                       f'Shapes are: {self.mask.shape}, {kernel_shape}')

    kernel = param_with_axes(
        'kernel', 
        self.kernel_init, 
        kernel_shape,
        self.param_dtype,
        axes=self.kernel_axes)

    if self.mask is not None:
      kernel *= self.mask

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      padding_lax = 'VALID'

    if self.use_bias:
      bias = param_with_axes(
          'bias', 
          self.bias_init, 
          (self.features,),
          self.param_dtype,
          axes=self.bias_axes)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    y = lax.conv_transpose(
        inputs,
        kernel,
        strides,
        padding_lax,
        rhs_dilation=self.kernel_dilation,
        precision=self.precision)

    if self.padding == 'CIRCULAR':
      # For circular padding, we need to identify the size of the final output
      # ("period") along each spatial dimension, pad each dimension to an
      # integer number of periods, and wrap the array periodically around each
      # dimension. Padding should be done in such a way that the start of the
      # original input data inside the padded array is located at integer
      # number of periods - otherwise the result would be circularly shifted.

      # Compute period along each spatial dimension - it's input size scaled
      # by the stride.
      scaled_x_dims = [
          x_dim * stride for x_dim, stride in zip(jnp.shape(inputs)[1:-1], strides)
      ]
      # Compute difference between the current size of y and the final output
      # size, and complement this difference to 2 * period - that gives how
      # much we need to pad.
      size_diffs = [
          -(y_dim - x_dim) % (2 * x_dim)
          for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)
      ]
      # Divide the padding equaly between left and right. The choice to put
      # "+1" on the left (and not on the right) represents a convention for
      # aligning even-sized kernels.
      total_pad = [
          ((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs
      ]
      y = jnp.pad(y, [(0, 0)] + total_pad + [(0, 0)])
      # Wrap the result periodically around each spatial dimension,
      # one by one.
      for i in range(1, y.ndim - 1):
        y = y.reshape(y.shape[:i] + (-1, scaled_x_dims[i - 1]) +
                      y.shape[i + 1:])
        y = y.sum(axis=i)

    if self.use_bias:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)

    return y