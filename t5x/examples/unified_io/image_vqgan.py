"""Code for pre-trained VQAGAN that tokenizes images"""
import math
from typing import Any

import jax

from t5x.examples.unified_io.config import VAEConfig

from flax import linen as nn
import jax.numpy as jnp

from t5x.examples.unified_io import layers

default_init = nn.initializers.lecun_normal()
zero_init =  nn.initializers.zeros
one_init = nn.initializers.ones

PyTreeDef = type(jax.tree_util.tree_structure(None))


class ResBlock(nn.Module):
  n_in: int
  n_out: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    h = x

    scale_init = one_init
    bias_init = zero_init

    h = layers.GroupNorm(
        use_bias = True,
        use_scale = True,
        bias_init = bias_init,
        scale_init = scale_init,
        name='norm1')(h)

    h = layers.nonlinearity(h)
    
    w_init = default_init
    b_init = zero_init

    h = layers.Conv(
      features=self.n_out,
      kernel_size=(3, 3),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv1')(h)

    h = layers.GroupNorm(
        use_bias = True,
        use_scale = True,
        bias_init = bias_init,
        scale_init = scale_init,
        name='norm2')(h)

    h = layers.nonlinearity(h)

    h = layers.Conv(
      features=self.n_out,
      kernel_size=(3, 3),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv2')(h)

    if self.n_in != self.n_out:
      x = layers.Conv(
        features=self.n_out,
        kernel_size=(1,1),
        dtype=self.dtype,
        kernel_init=w_init,
        use_bias=True,
        bias_init=b_init,
        kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
        bias_axes=('axis_3',),
        name='nin_shortcut')(x)
    return x + h


class AttnBlock(nn.Module):
  n_in: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    h_ = x

    scale_init = one_init
    bias_init = zero_init

    h_ = layers.GroupNorm(
        use_bias = True,
        use_scale = True,
        bias_init = bias_init,
        scale_init = scale_init,
        name='norm')(h_)

    w_init = default_init
    b_init = zero_init

    q = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='q')(h_)

    k = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='k')(h_)

    v = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='v')(h_)

    b, h, w, c = q.shape

    w_ = jnp.einsum('bqc,bkc->bqk', jnp.reshape(q, (b, h*w, c)), jnp.reshape(k, (b, h*w, c)))
    w_ = w_ * (c ** -0.5)
    w_ = jax.nn.softmax(w_).astype(self.dtype)
    h_ = jnp.einsum('bqk,bkd->bqd', w_, jnp.reshape(v, (b, h*w, c)))
    h_ = jnp.reshape(h_, (b, h, w, c))

    h_ = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_init=w_init,
      use_bias=True,
      bias_init=b_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='proj_out')(h_)

    return x+h_


class Downsample(nn.Module):
  n_in: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    x = layers.Conv(
      features=self.n_in,
      kernel_size=(3, 3),
      strides=(2,2),
      dtype=self.dtype,
      kernel_init=default_init,
      use_bias=True,
      bias_init=zero_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv')(x)

    return x


class Upsample(nn.Module):
  n_in: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    B, H, W, C = x.shape
    x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method='nearest')

    x = layers.Conv(
      features=self.n_in,
      kernel_size=(3, 3),
      strides=(1,1),
      dtype=self.dtype,
      kernel_init=default_init,
      use_bias=True,
      bias_init=zero_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv')(x)

    return x


class VAE_Encoder(nn.Module):
  """Jax implementation of Taming VAE encoder"""
  config: VAEConfig

  @nn.compact
  def __call__(self, x, training=False):
    cfg = self.config
    curr_res = cfg.resolution
    num_resolutions = len(cfg.ch_mult)
    in_ch_mult = (1,)+tuple(cfg.ch_mult)

    hs = layers.Conv(
      features=1 * cfg.ch,
      kernel_size=(3, 3),
      strides=(1, 1),
      dtype=cfg.dtype,
      kernel_init=default_init,
      use_bias=True,
      bias_init=zero_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_in')(x)

    for i_level in range(num_resolutions):
      block_in = cfg.ch * in_ch_mult[i_level]
      block_out = cfg.ch * cfg.ch_mult[i_level]

      for i_block in range(cfg.num_res_blocks):
        hs = ResBlock(
          block_in,
          block_out,
          cfg.dtype,
          name=f"down_{i_level}_block_{i_block}")(hs)

        block_in = block_out
        if curr_res in cfg.attn_resolutions:
          hs = AttnBlock(
            block_in,
            name=f"down_{i_level}_attn_{i_block}")(hs)

      if i_level != num_resolutions-1:
        hs = Downsample(
          block_in,
          name=f"down_{i_level}_downsample")(hs)
        curr_res = curr_res // 2

    hs = ResBlock(
        block_in, 
        block_in, 
        name='mid_block_1')(hs)
    
    hs = AttnBlock(
        block_in, 
        name='mid_attn_1')(hs)

    hs = ResBlock(
        block_in, 
        block_in, 
        name='mid_block_2')(hs)

    hs = layers.GroupNorm(
        use_bias = True,
        use_scale = True,
        name='norm_out')(hs)

    hs = layers.nonlinearity(hs)

    hs = layers.Conv(
      features=cfg.z_channels,
      kernel_size=(3, 3),
      dtype=cfg.dtype,
      kernel_init=default_init,
      use_bias=True,
      bias_init=zero_init,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_out')(hs)

    return hs


class VAE_Decoder(nn.Module):
  """Jax implementation of Taming VAE encoder"""
  config: VAEConfig
  params_init: Any = None

  @nn.compact
  def __call__(self, x, training=False):

    cfg = self.config
    num_resolutions = len(cfg.ch_mult)
    curr_res = cfg.resolution // 2**(num_resolutions-1)
    block_in = cfg.ch*cfg.ch_mult[num_resolutions-1]

    # z to block_in
    h = layers.Conv(
      features=block_in,
      kernel_size=(3, 3),
      strides=(1, 1),
      dtype=cfg.dtype,
      use_bias=True,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_in')(x)

    h = ResBlock(
        block_in, 
        block_in, 
        name='mid_block_1')(h)

    h = AttnBlock(
      block_in, 
      name='mid_attn_1')(h)
    
    h = ResBlock(
        block_in, 
        block_in, 
        name='mid_block_2')(h)

    for i_level in reversed(range(num_resolutions)):
      i_idx = num_resolutions - i_level-1
      block_out = cfg.ch * cfg.ch_mult[i_level]
      for i_block in range(cfg.num_res_blocks+1):
        h = ResBlock(
            block_in, 
            block_out,
            name=f"up_{i_idx}_block_{i_block}")(h)
        block_in = block_out
        if curr_res in cfg.attn_resolutions:
          h = AttnBlock(
              block_in, 
              name=f"up_{i_idx}_attn_{i_block}")(h)
      if i_level != 0:
        h = Upsample(
            block_in, 
            name=f"up_{i_idx}_upsample")(h)
        curr_res = curr_res * 2
    
    h = layers.GroupNorm(
        use_bias = True,
        use_scale = True,
        name='norm_out')(h)

    h = layers.nonlinearity(h)

    h = layers.Conv(
      features=cfg.out_ch,
      kernel_size=(3, 3),
      strides=(1, 1),
      dtype=cfg.dtype,
      use_bias=True,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_out')(h)

    return h


class ImageDVQGAN(nn.Module):
  """Jax implementation of Taming Transformers VQGAN"""
  # https://github.com/CompVis/taming-transformers
  config: VAEConfig

  def setup(self):
    cfg = self.config
    self.encoder = VAE_Encoder(cfg)
        
    self.quant_conv = layers.Conv(
      features=cfg.z_channels,
      kernel_size=(1, 1),
      dtype=cfg.dtype,
      use_bias=True,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='quant_conv')

    self.quantize = layers.VectorQuantizer(
      cfg.n_embed,
      cfg.z_channels,
      beta=0.25,
    )

    self.post_quant_conv = layers.Conv(
      features=cfg.z_channels,
      kernel_size=(1, 1),
      dtype=cfg.dtype,
      use_bias=True,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='post_quant_conv')

    self.decoder = VAE_Decoder(cfg)

  def encode(self, x, training=False):
    h = self.encoder(x, training)

    h = self.quant_conv(h)
    quant, emb_loss, info = self.quantize(h)
    return quant, emb_loss, info

  def decode(self, quant, training=False):
    quant = self.post_quant_conv(quant)
    dec = self.decoder(quant, training)
    return dec

  def decode_code(self, code_b):
    quant_b = self.quantize.get_codebook_entry(code_b)
    bs, seq_len, dim = quant_b.shape
    size = int(math.sqrt(seq_len))
    quant_b = jnp.reshape(quant_b, (bs, size, size, dim))
    dec = self.decode(quant_b)
    return dec

  def get_codebook_indices(self, x, vae_decode=False, training=False):
    h = self.encoder(x, training)
    h = self.quant_conv(h)
    z, _, [_, _, indices] = self.quantize(h)

    if vae_decode:
      _ = self.decode(z, training)

    return jnp.reshape(indices, (jnp.shape(h)[0], -1))

  @nn.compact
  def __call__(self, x, training=False):

    quant, diff, _ = self.encode(x, training)
    dec = self.decode(quant, training)
    return dec
