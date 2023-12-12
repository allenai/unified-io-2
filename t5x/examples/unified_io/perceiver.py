"""Perceiver Resampler model used in the history encoders"""
from typing import Callable, Iterable, Union

import jax

from t5x.examples.unified_io import layers
from t5x.examples.unified_io.config import ImageResamplerConfig, AudioResamplerConfig

PyTreeDef = type(jax.tree_util.tree_structure(None))
from flax import linen as nn
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]
default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


class CrossAttention(nn.Module):
  """Cross-attention layer."""
  config: Union[ImageResamplerConfig, AudioResamplerConfig]
  droppath_rate: float = 0.0

  @nn.compact
  def __call__(self, latents, context, mask=None, deterministic=False):
    cfg = self.config

    # Cross attention block.
    assert context.ndim == 3
    assert latents.ndim == 3
    assert latents.shape[-1] == context.shape[-1]

    # q: latents. [batch, latent_length, emb_dim]
    # kv: context. [batch, context_length, emb_dim]
    inputs_q = layers.LayerNorm(dtype=cfg.dtype, name='pre_xattention_layer_norm')(latents)
    inputs_kv = context

    # Cross-attention
    # [batch, latent_length, emb_dim] x [batch, context_length, emb_dim]
    # => [batch, latent_length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        dropout_broadcast_dims=cfg.dropout_broadcast_dims,
        float32_logits=cfg.float32_attention_logits,
        qk_norm=cfg.xattn_qk_norm,
        clip_attn_logit=cfg.clip_attn_logit,
        scaled_cosine=cfg.xattn_scaled_cosine,
        name='xattention')(
            inputs_q, inputs_kv, mask, deterministic=deterministic)

    x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=cfg.dropout_broadcast_dims)(x, deterministic=deterministic)

    x = layers.DropPath(rate=self.droppath_rate)(x, deterministic=deterministic) + latents

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dropout_broadcast_dims=cfg.dropout_broadcast_dims,
        dtype=cfg.dtype,
        name='mlp',
    )(y, deterministic=deterministic)

    y = layers.DropPath(rate=self.droppath_rate)(y, deterministic=deterministic) + x
    return y


class Attention(nn.Module):
  """Self-attention layer."""
  config: Union[ImageResamplerConfig, AudioResamplerConfig]
  droppath_rate: float = 0.0

  @nn.compact
  def __call__(self, latents, mask=None, deterministic=False):
    cfg = self.config

    # qkv: latents. [batch, latent_length, emb_dim]
    x = layers.LayerNorm(dtype=cfg.dtype, name='pre_attention_layer_norm')(latents)

    # Self-attention
    # [batch, latent_length, emb_dim]
    # => [batch, latent_length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        dropout_broadcast_dims=cfg.dropout_broadcast_dims,
        float32_logits=cfg.float32_attention_logits,
        qk_norm=cfg.attn_qk_norm,
        clip_attn_logit=cfg.clip_attn_logit,
        scaled_cosine=cfg.attn_scaled_cosine,
        name='attention')(
            x, x, mask, deterministic=deterministic)

    x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=cfg.dropout_broadcast_dims)(x, deterministic=deterministic)

    x = layers.DropPath(rate=self.droppath_rate)(x, deterministic=deterministic) + latents

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dropout_broadcast_dims=cfg.dropout_broadcast_dims,
        dtype=cfg.dtype,
        name='mlp',
    )(y, deterministic=deterministic)

    y = layers.DropPath(rate=self.droppath_rate)(y, deterministic=deterministic) + x
    return y


class PerceiverResampler(nn.Module):
  """Perceiver resampler: a stack of cross-attention layers."""
  config: Union[ImageResamplerConfig, AudioResamplerConfig]

  def setup(self):
    cfg = self.config
    self.latents = layers.param_with_axes(
      'resampler_latents',
      default_embed_init,
      (cfg.latents_size, cfg.emb_dim),
      jnp.float32,
      axes=(('image_patch', 'embed')))

  @nn.compact
  def __call__(self, embed, *, mask=None, deterministic=False):
    cfg = self.config
    bs, seq_len, dim = embed.shape
        
    if mask is None:
      mask = jnp.ones([bs, seq_len], dtype=jnp.int32)
    
    embed = embed.reshape((bs, seq_len, dim))
    query_mask = jnp.ones([bs, cfg.latents_size], dtype=mask.dtype)
    key_mask = mask.reshape((bs, seq_len))
    latents = jnp.expand_dims(self.latents, axis=0)
    latents = jnp.tile(latents, [bs, 1, 1]).astype(cfg.dtype)

    embed = layers.LayerNorm(dtype=cfg.dtype, name='context_norm')(embed)
    xattention_mask = layers.make_attention_mask(query_mask, key_mask, dtype=cfg.dtype)
    attention_mask = layers.make_attention_mask(query_mask, query_mask, dtype=cfg.dtype)
    
    dpr = [x for x in np.linspace(0, cfg.droppath_rate, cfg.num_layers)]
    for lyr in range(cfg.num_layers):
      if lyr in cfg.xattention_index:
        latents = CrossAttention(
          config=cfg, droppath_rate=dpr[lyr],
          name=f'layers_{lyr}')(latents, embed, xattention_mask, deterministic)
      else:
        latents = Attention(
          config=cfg, droppath_rate=dpr[lyr],
          name=f'layers_{lyr}')(latents, attention_mask, deterministic)

    latents = layers.LayerNorm(dtype=cfg.dtype, name='perceiver_norm')(latents)
    
    return latents


class LinearResampler(nn.Module):
  """Perceiver resampler: a stack of cross-attention layers."""
  config: Union[ImageResamplerConfig, AudioResamplerConfig]

  def setup(self):
    cfg = self.config

    self.time_emb = layers.param_with_axes(
      'time_embedding',
      default_embed_init,
      (cfg.max_frames, cfg.emb_dim),
      jnp.float32,
      axes=(('axis_0', 'embed')))

  @nn.compact
  def __call__(self, embed, *, mask=None, deterministic=False):
    cfg = self.config
    bs, times, seq_len, dim = embed.shape

    if dim != cfg.emb_dim:
      embed = layers.DenseGeneral(
        cfg.emb_dim, dtype=cfg.dtype,
        kernel_axes=('embed', 'mlp'),
      )(embed)
      dim = cfg.emb_dim

    time_emb = self.time_emb[:times].reshape((times, 1, dim)).astype(cfg.dtype)
    
    embed = embed + time_emb
    
    embed = jnp.transpose(embed, (0, 2, 1, 3)).reshape((bs, seq_len, times*dim))

    embed = layers.DenseGeneral(
      cfg.emb_dim, dtype=cfg.dtype,
      kernel_axes=('image_patch', 'embed'),
    )(embed)

    return embed


class Resampler(nn.Module):
  """Perceiver resampler: a stack of cross-attention layers."""
  config: Union[ImageResamplerConfig, AudioResamplerConfig]

  @nn.compact
  def __call__(self, embed, *, mask=None, deterministic=False):
    cfg = self.config
    embed = PerceiverResampler(cfg)(embed, mask=mask, deterministic=deterministic)
    return embed