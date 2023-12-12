"""Code for pre-trained image encoder the turn images into features"""
import functools
import math
import random
from typing import Any, Callable, Iterable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax

import t5x.examples.unified_io.layers as layers
from t5x.examples.unified_io.config import ImageVitFeatureConfig, AudioVitFeatureConfig
from t5x.examples.unified_io.layers import DenseGeneral

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_kernel_init = nn.initializers.glorot_uniform()
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


def QuickGELU(x): return x * nn.sigmoid(1.702 * x)


class MLP(nn.Module):
  config: Union[ImageVitFeatureConfig, AudioVitFeatureConfig]
  param_dict: Any = None

  @nn.compact
  def __call__(self, x):
    
    cfg = self.config
    kernel_init = nn.initializers.glorot_uniform() \
        if self.param_dict is None \
        else lambda *_ : jnp.transpose(jnp.array(self.param_dict['c_fc']['weight']), (1,0))
    
    bias_init = nn.initializers.zeros if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['c_fc']['bias'])

    x = DenseGeneral(
      cfg.mlp_dim,
      dtype=cfg.dtype,
      use_bias=True,
      kernel_init=kernel_init,
      bias_init=bias_init,
      kernel_axes=('embed', 'mlp'),
      bias_axes=('mlp',),
      name='c_fc',
    )(x)

    x = jax.nn.gelu(x, approximate=False)
    
    kernel_init = nn.initializers.glorot_uniform() \
        if self.param_dict is None \
        else lambda *_ : jnp.transpose(jnp.array(self.param_dict['c_proj']['weight']), (1,0))
    
    bias_init = nn.initializers.zeros if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['c_proj']['bias'])
    
    x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))

    x = DenseGeneral(
      cfg.emb_dim,
      dtype=cfg.dtype,
      use_bias=True,
      kernel_init=kernel_init,
      bias_init=bias_init,
      kernel_axes=('mlp', 'embed'),
      bias_axes=('embed',),
      name='c_proj',
    )(x)

    return x
    

def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False):
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

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)# * depth ** -0.5

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)
  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))

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
  dropout_rate: float = 0.
  kernel_init: Initializer = default_kernel_init
  params_init: Any = None # paramter intialization to pass into the multi-head attention. 
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
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.
    Returns:
      output of shape `[batch, length, q_features]`.
    """

    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        use_bias=True,
        kernel_axes=('embed', 'joined_kv'),
        bias_axes=('joined_kv',),
        dtype=self.dtype)

    if self.params_init is not None:
      qkv_kernel = jnp.split(jnp.transpose(np.array(self.params_init['in_proj_weight']), (1,0)), 3, axis=1)
      qkv_bias = jnp.split(np.array(self.params_init['in_proj_bias']), 3, axis=0)

    query_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[0])
    query_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(qkv_bias[0])

    key_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[1])
    key_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(qkv_bias[1])

    value_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.array(qkv_kernel[2])
    value_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(qkv_bias[2])

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_kernel_init, bias_init=query_bias_init, name='query')(inputs_q)
    key = projection(kernel_init=key_kernel_init,  bias_init=key_bias_init, name='key')(inputs_kv)
    value = projection(kernel_init=value_kernel_init, bias_init=value_bias_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

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
      attention_bias = layers.combine_biases(attention_bias, bias, abs_bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits)

    out_kernel_init = self.kernel_init if self.params_init is None else lambda *_ : jnp.transpose(jnp.array(self.params_init['out_proj']['weight']), (1,0))
    out_bias_init = nn.initializers.zeros if self.params_init is None else lambda *_ : jnp.array(self.params_init['out_proj']['bias'])
    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        use_bias=True,
        kernel_init=out_kernel_init,
        bias_init=out_bias_init,
        kernel_axes=('joined_kv', 'embed'),
        bias_axes=('embed',),
        dtype=self.dtype,
        name='out')(
            x)

    return out


class ResidualAttentionBlock(nn.Module):
  config: Union[ImageVitFeatureConfig, AudioVitFeatureConfig]
  param_dict: Any = None

  @nn.compact
  def __call__(self,
               inputs,
               mask,
               *,
               enable_dropout: bool = True,
               ):
               
    cfg = self.config
    bias_init = nn.initializers.zeros \
        if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['ln_1']['bias'])

    scale_init = nn.initializers.ones \
      if self.param_dict is None \
      else lambda *_ : jnp.array(self.param_dict['ln_1']['weight'])

    x = layers.LayerNormWithBias(
        epsilon=1e-5,
        bias_init=bias_init, 
        scale_init=scale_init,
        dtype=cfg.dtype, 
        name='ln_1')(inputs)
    
    attn_init = None if self.param_dict is None else self.param_dict['attn']
    x = MultiHeadDotProductAttention(
        num_heads = cfg.num_heads,
        head_dim = cfg.head_dim,
        dtype = cfg.dtype,
        dropout_rate = cfg.dropout_rate,
        params_init = attn_init)(x, x, mask=mask) + inputs

    bias_init = nn.initializers.zeros \
        if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['ln_2']['bias'])

    scale_init = nn.initializers.ones \
      if self.param_dict is None \
      else lambda *_ : jnp.array(self.param_dict['ln_2']['weight'])

    y = layers.LayerNormWithBias(
        epsilon=1e-5,
        bias_init=bias_init, 
        scale_init=scale_init,
        dtype=cfg.dtype, 
        name='ln_2')(x)

    mlp_dict = None if self.param_dict is None else self.param_dict['mlp']
    y = MLP(cfg, mlp_dict)(y) + x
    return y


class Transformer(nn.Module):
  config: Union[ImageVitFeatureConfig, AudioVitFeatureConfig]
  param_dict: Any = None

  @nn.compact
  def __call__(self,
               x, 
               mask = None,
               *,
               enable_dropout: bool = True,
               ):
    cfg = self.config
    xs = []
    for _ in range(cfg.num_layers):
      resblocks_dict_i = None if self.param_dict is None else self.param_dict['resblocks'][str(_)]
      x = ResidualAttentionBlock(cfg, param_dict=resblocks_dict_i)(x, mask=mask)
      xs.append(x)
      
    return x, xs


class VisionTransformer(nn.Module):
  config: Union[ImageVitFeatureConfig, AudioVitFeatureConfig]
  param_dict: Any = None

  def setup(self):
    cfg = self.config

    scale = cfg.emb_dim ** -0.5
    kernel_init = nn.initializers.normal(stddev=scale) if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['class_embedding'])
    
    self.class_embedding = param_with_axes(
        'class_embedding',
        kernel_init, 
        (cfg.emb_dim, ),
        jnp.float32, 
        axes=(('embed',)))

    kernel_init = nn.initializers.normal(stddev=scale) if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['positional_embedding'])
    
    self.positional_embedding = param_with_axes(
        'positional_embedding',
        kernel_init, 
        (cfg.num_pos, cfg.emb_dim),
        jnp.float32, 
        axes=(('axis_0', 'embed')))

  def add_pos_emb(self, x, pos_ids, patch_num):
    cls_emb = self.positional_embedding[0:1]
    pos_emb = self.positional_embedding[1:]

    pos_emb = jnp.reshape(pos_emb, 
        (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1]))
    
    (patch_num_0, patch_num_1) = patch_num
    if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
      pos_emb = jax.image.resize(pos_emb, (patch_num_0, patch_num_1, pos_emb.shape[-1]), "bicubic")

    pos_emb = jnp.reshape(pos_emb, [-1, pos_emb.shape[-1]])[pos_ids]
    x = x + jnp.concatenate([jnp.tile(cls_emb[None,:,:], (x.shape[0], 1, 1)), pos_emb], axis=1)
    return x

  @nn.compact
  def __call__(self,
               x,
               mask,
               pos_ids,
               *,
               enable_dropout: bool = True,
               patch_num: Any = (16, 16),
               ):
    cfg = self.config
    
    B = x.shape[0]
    kernel_init = nn.initializers.glorot_uniform()

    x = layers.DenseGeneral(
        features=cfg.emb_dim,
        use_bias=False,
        kernel_init=kernel_init,
        dtype=cfg.dtype,
        kernel_axes=('axis_0', 'embed'),
        name='embedding')(
            x)

    x = jnp.concatenate([
        jnp.repeat(self.class_embedding[None, None, :], B, axis=0),
        x], axis=1)

    mask = jnp.concatenate((jnp.ones([B, 1], dtype=jnp.int32), mask), axis=1)
    
    x = self.add_pos_emb(x, pos_ids, patch_num)    
    
    bias_init = nn.initializers.zeros \
        if self.param_dict is None \
        else lambda *_ : jnp.array(self.param_dict['ln_pre']['bias'])

    scale_init = nn.initializers.ones \
      if self.param_dict is None \
      else lambda *_ : jnp.array(self.param_dict['ln_pre']['weight'])

    x = layers.LayerNormWithBias(
        epsilon=1e-5,
        bias_init=bias_init, 
        scale_init=scale_init,
        dtype=cfg.dtype, 
        name='pre_ln')(x)

    attn_mask = layers.make_attention_mask(mask, mask, dtype=cfg.dtype)
    transformer_dict = None if self.param_dict is None else self.param_dict['transformer']
  
    x, xs = Transformer(cfg, param_dict=transformer_dict)(x, attn_mask)
  
    # remove the cls token
    x = x[:,1:,:]

    x1 = xs[1][:,1:,:]
    
    return x, x1


class ImageEncoder(nn.Module):
  """Builds features from an image"""
  config: Union[ImageVitFeatureConfig, AudioVitFeatureConfig]

  def setup(self):
    cfg = self.config
    self.vision_transformer = VisionTransformer(config=cfg, param_dict=None)

  @nn.compact
  def __call__(self, x, mask, pos_ids, *, enable_dropout: bool = True, patch_num: Any = (16, 16)):
    x, x1 = self.vision_transformer(x, mask, pos_ids, enable_dropout=enable_dropout, patch_num=patch_num)
    return x, x1
    