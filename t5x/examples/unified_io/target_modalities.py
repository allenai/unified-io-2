"""Defines `ModalityEncoder` for target modalities"""
from dataclasses import dataclass
from typing import Dict

import jax
import numpy as np
import seqio
import tensorflow as tf
from flax import linen as nn

from t5x.examples.unified_io import layers, seq_features
from t5x.examples.unified_io.config import *
from t5x.examples.unified_io.data.data_utils import trim_or_pad_tf, get_default_vocabulary
from t5x.examples.unified_io.image_vqgan import ImageDVQGAN
from t5x.examples.unified_io.audio_vqgan import ASTVQGAN
from t5x.examples.unified_io.input_modalities import ModalityEncoder
from t5x.examples.unified_io.layers import param_with_axes
from t5x.examples.unified_io.seq_features import TargetSequence


TEXT_MODALITY_INDEX = 0
IMAGE_MODALITY_INDEX = 1
AUDIO_MODALITY_INDEX = 2


class TextEmbedder(nn.Module):
  config: T5Config
  embedding_layer: nn.Module

  def setup(self):
    cfg = self.config
    self.pos_emb_cache = layers.get_1d_position_embedding(
      cfg.text_pos_emb, cfg.decoder_max_text_length, cfg.emb_dim, cfg.head_dim, True, 1, cfg.dtype)

  @nn.compact
  def __call__(self, inputs, mask=None, pos_ids=None, segment_ids=None, 
              targets=None, init=False, decode=False, decode_length=None, 
              cur_index=None):

    cfg = self.config
    bs = inputs.shape[0]
    if mask is None:
      mask = (inputs > 0).astype(jnp.int32)
    if pos_ids is None:
      if decode_length is None:
        decode_length = inputs.shape[1]
      pos_ids = jnp.arange(decode_length, dtype=jnp.int32)[None, ...]
    
    x = self.embedding_layer(inputs)

    if cur_index is None:
      pos_emb = self.pos_emb_cache[None,:,:][jnp.arange(bs)[:, None], pos_ids]
    else:
      pos_emb = jax.lax.dynamic_slice(self.pos_emb_cache, (cur_index, 0), (1, self.pos_emb_cache.shape[-1]))
      pos_emb = jnp.tile(pos_emb[None,:,:], (bs, 1, 1))

    if "rope" not in cfg.text_pos_emb:
      x += pos_emb      

    if "llama_rope" in cfg.text_pos_emb:
      modality_emb = param_with_axes(
        "modality_embedding",
        nn.initializers.normal(stddev=0.02),
        (cfg.emb_dim,),
        axes=(('embed',)),
      )

      x += modality_emb[None, None, :].astype(cfg.dtype)

    attn_pattern_mask = jnp.ones([bs, 4, x.shape[1], x.shape[1]], cfg.dtype)

    return TargetSequence(
      x, pos_emb, jnp.array(TEXT_MODALITY_INDEX), mask, attn_pattern_mask=attn_pattern_mask,
      subsegments=segment_ids, target_tokens=targets, loss_mask=mask
    )


class BasicDecoder(nn.Module):
  vocab_size: int
  config: T5Config
  embedding_layer: nn.Module

  @nn.compact
  def __call__(self, x, decode=False):
    cfg = self.config
    if cfg.logits_via_embedding:
      logits = self.embedding_layer.attend(x)
      logits = logits / jnp.sqrt(x.shape[-1])
    else:
      logits = layers.DenseGeneral(
          self.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense')(
              x)
    return logits


class TargetTextEncoder(ModalityEncoder):
  """Tokenize and embed target text, handles multiple target texts"""

  def preprocess_inputs(self, features, output_features, sequence_length) -> Dict:
    vocab = output_features[f'targets/text/tokens'].vocabulary

    text_targets = features.get(f"text_targets")
    if text_targets is None:
      # TODO maybe this should completely empty?
      text_targets = vocab.encode_tf('')

    max_len = sequence_length[f"text_targets"]
    if text_targets.dtype == tf.dtypes.string:
      text_targets = vocab.encode_tf(text_targets)

    text_targets = text_targets[..., :max_len-1]

    if isinstance(text_targets, tf.RaggedTensor):
      inputs = seqio.preprocessors._append_to_innermost_axis(text_targets, vocab.eos_id)
      row_length = inputs.row_lengths()
      position_ids = tf.cast(tf.ragged.range(row_length).flat_values, tf.int32)
      segment_ids = tf.cast(inputs.value_rowids()+1, tf.int32)
      text_targets = tf.cast(inputs.flat_values, tf.int32)
    else:
      if tf.shape(text_targets)[0] == 0 or text_targets[-1] != vocab.eos_id:
        text_targets = tf.pad(text_targets, paddings=[[0, 1]], constant_values=vocab.eos_id)
      text_len = tf.shape(text_targets)[0]
      position_ids = tf.range(text_len, dtype=tf.int32)
      segment_ids = tf.ones((text_len,), dtype=tf.int32)

    return dict(
      tokens=text_targets,
      pos_ids=position_ids,
      segment_ids=segment_ids
    )

  def convert_choices(self, choices, sequence_length):
    text_len = sequence_length.get("text_targets")
    if text_len is None:
      text_len = sequence_length["targets/text/tokens"]
    vocab = get_default_vocabulary()
    if choices.dtype != tf.int32:
      choices = vocab.encode_tf(choices)

    if isinstance(choices, tf.RaggedTensor):
      choices = choices[..., :text_len-1]
      inputs = seqio.preprocessors._append_to_innermost_axis(choices, vocab.eos_id)
      inputs = inputs.to_tensor()
    else:
      # `choice` is already padded, so the pre-processor should have added EOS already
      # I think this only applies to GRIT which has to handle variable lengths options per example
      has_eos = tf.reduce_any(choices == 1, axis=-1)
      all_zero = tf.reduce_all(choices == 0, axis=-1)
      tf.assert_equal(tf.reduce_all(tf.logical_or(has_eos, all_zero)), True)
      inputs = choices
    inputs = tf.pad(inputs, [[0, 0], [0, text_len - tf.shape(inputs)[1]]], constant_values=vocab.pad_id)
    inputs = tf.ensure_shape(inputs, [None, text_len])
    return inputs

  def convert_inputs(self, features, sequence_length) -> Dict:
    # Support old style and new style sequence_lengths
    voc = get_default_vocabulary()
    text_len = sequence_length.get("text_targets")
    if text_len is None:
      text_len = sequence_length["targets/text/tokens"]
    # vocab = get_default_vocabulary()
    for k, v in features.items():
      # TODO trimming here is a bit questionable since it might trim EOS, trimming
      # should really happen between tokenization and appending EOS, but keep for now
      # since older versions did this too
      features[k] = trim_or_pad_tf(v, text_len, pad_constant=voc.pad_id)
    tokens = features.pop("tokens")
    features["targets"] = tokens

    features["inputs"] = make_autoregressive_inputs(
      tokens, sequence_id=features.get("segment_ids"), bos_id=voc.pad_id
    )

    # remove the end token mask here, assume eos_id is larger than pad_id
    if tf.math.reduce_sum(tokens) > voc.eos_id:   # > 1
      features["mask"] = tf.cast(tokens > voc.pad_id, tf.int32)
    else:
      features["mask"] = tf.cast(tf.zeros(tokens.shape), tf.int32)
    return features

  def get_output_features(self) -> Dict[str, seqio.Feature]:
    return {
      "tokens": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True),
      "pos_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
      "segment_ids": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
    }

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return TextEmbedder(config, shared_embedding)

  def get_decoder(self, config: Config, shared_embedding) -> nn.Module:
    return BasicDecoder(config.vocab_size, config, shared_embedding)


def _shift_right_by_one(tensor: tf.Tensor, bos_id: int = 0) -> tf.Tensor:
  """Shift the input tensor to the right by one position without wrapping."""

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(f"Only numeric types are supported. Got: {tensor.dtype}")
  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=0)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  depth = tf.shape(tensor)[0]
  mask = tf.one_hot(0, depth=depth, on_value=0, off_value=1, dtype=tensor.dtype)

  # Expand dims of mask to broadcast to rolled.
  dim_expansion = [slice(None, None)] + [None] * (len(rolled.shape) - 1)
  mask = mask[dim_expansion]
  return rolled * mask + (1 - mask) * bos_id


# support customized bos_id
def make_autoregressive_inputs(
  targets: tf.Tensor,
  sequence_id: tf.Tensor = None,
  output_dtype: Optional[tf.dtypes.DType] = None,
  bos_id: int = 0,
) -> tf.Tensor:
  """Generate inputs for an autoregressive model, by shifting the targets.

  Modified from mesh_tensorflow.transformer.transformer.autoregressive_inputs.

  For the first element of each sequence, the returned input id is 0.

  For a "packed" dataset, also pass the sequence_id tensor, which aligns
  with the targets tensor and contains different values for different
  concatenated examples.

  Example for a packed dataset:

  ```
        targets = [3, 8, 1, 9, 1, 5, 4, 1, 0, 0]
    sequence_id = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [0, 3, 8, 0, 9, 0, 5, 4, 0, 0]
                            |     |        |
                            These positions are set to 0 if sequence_id is not
                            None.
  ```

  Args:
    targets: a tf.int32 tensor with shape [length].
    sequence_id: an optional tensor with the same shape as targets.
    output_dtype: an optional output data type.
    bos_id: bos id.

  Returns:
    a tensor with dtype tf.int32 and the same shape as targets.
  """
  output_dtype = output_dtype or targets.dtype
  if sequence_id is not None and not sequence_id.dtype.is_integer:
    raise ValueError(
        "The sequence_id should be integer-valued tensors for a packed dataset."
    )
  if sequence_id is not None and len(targets.shape) > 1:
    raise ValueError(
        "Only 1-D sequences are supported with packing. Got a "
        f"packed {len(targets.shape)}-D sequence."
    )

  inputs = _shift_right_by_one(targets, bos_id)
  if inputs.dtype != output_dtype:
    inputs = tf.cast(inputs, output_dtype)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
        sequence_id, _shift_right_by_one(sequence_id)
    )
    not_first_in_sequence = tf.cast(not_first_in_sequence, output_dtype)
    first_ids = tf.cast((1 - not_first_in_sequence) * bos_id, output_dtype)
    inputs = inputs * not_first_in_sequence + first_ids
  return inputs


def _init_mask(height, width, is_bool_mask=False):
  attn_size = height * width
  mask = np.tril(np.ones([attn_size, attn_size], np.bool if is_bool_mask else np.float32))
  return mask


def get_row_mask(height=32, width=32, is_bool_mask=False):
  mask = _init_mask(height, width, is_bool_mask=is_bool_mask)
  step = width + 1
  for col in range(mask.shape[1]):
      mask[col + step:, col] = False if is_bool_mask else 0.0
  return mask  


def get_col_mask(height=32, width=32, is_bool_mask=False):
  mask = _init_mask(height, width, is_bool_mask=is_bool_mask)
  step = width - 1
  for col in range(mask.shape[1]):
      for i in range(1, mask.shape[0], step+1):
          mask[col + i: col + i + step, col] = False if is_bool_mask else 0.0
  return mask


def get_conv_mask(height=32, width=32, kernel=11, is_bool_mask=False):
    mask = _init_mask(height, width, is_bool_mask=is_bool_mask)
    shift = kernel // 2
    for pos in range(mask.shape[1]):
        mask[pos+1:, pos] = False if is_bool_mask else 0.0
        img = np.zeros([height, width])
        pixel_id = pos
        row = pixel_id // width
        col = pixel_id % width
        for r in range(-shift, shift+1):
            for c in range(-shift, shift+1):
                c_abs = max(min(c + col, width - 1), 0)
                r_abs = max(min(r + row, height - 1), 0)
                img[r_abs, c_abs] = 0.2
                cell_id = r_abs * width + c_abs
                if  cell_id > pos:
                    mask[cell_id, pos] = True if is_bool_mask else 1.0
        img[row, col] = 1.0
    return mask


class ImageViTVQGAN(nn.Module):
  config: T5Config
  vae_config: ImageViTVQGANConfig
  embedding_layer: nn.Module

  def setup(self):
    cfg = self.config
    vae_cfg = self.vae_config

    self.grid_size = [
        self.config.default_image_size[0] // self.vae_config.patch_size[0],
        self.config.default_image_size[1] // self.vae_config.patch_size[1],
    ]
    
    if cfg.image_tokenizer_type == 'vqgan':
      self.discrete_vae = ImageDVQGAN(self.vae_config)
    else:
      raise NotImplementedError(cfg.image_tokenizer_type)

    assert cfg.emb_dim == self.embedding_layer.embedding.shape[-1]

    # construct the row, col and conv mask.
    row_mask = get_row_mask(self.grid_size[0], self.grid_size[1])
    col_mask = get_col_mask(self.grid_size[0], self.grid_size[1])
    conv_mask = get_conv_mask(self.grid_size[0], self.grid_size[1])
    full_mask = _init_mask(self.grid_size[0], self.grid_size[1])
    
    self.attn_mask = jnp.stack([
        jnp.array(row_mask, cfg.dtype), 
        jnp.array(col_mask, cfg.dtype),
        jnp.array(conv_mask, cfg.dtype), 
        jnp.array(full_mask, cfg.dtype),
    ])

    self.pos_emb_cache = layers.get_2d_position_embedding(
        cfg.image_pos_emb, vae_cfg.default_input_size, vae_cfg.patch_size,
        cfg.emb_dim, cfg.head_dim, 2, cfg.dtype)

  def target_image_to_seq(self, image, loss_mask=None, init=False, 
                          task_mask=None):
    cfg = self.config
    bs = image.shape[0]

    if init: _ = self.discrete_vae(image)
    target_tokens = self.discrete_vae.get_codebook_indices(image)
    
    target_tokens = target_tokens + 2
    target_tokens = jax.lax.stop_gradient(target_tokens)

    if task_mask is not None and cfg.unk_mask_token:
      input_tokens = (
        target_tokens * (1 - task_mask.astype(jnp.int32)) +
        jnp.ones(target_tokens.shape, dtype=jnp.int32) * task_mask.astype(jnp.int32)
      )
    else:
      input_tokens = target_tokens

    input_tokens = jnp.concatenate([
      jnp.zeros((input_tokens.shape[0], 1), dtype=jnp.int32),
      input_tokens], axis=1)
    
    if cfg.shift_back_input_token: 
      if task_mask is not None:
        start_idx = (task_mask.sum(axis=-1) > 0).astype(jnp.int32)
      else:
        start_idx = jnp.zeros((bs,), dtype=jnp.int32)
      pos_ids = jnp.arange(target_tokens.shape[1])[None, :] + start_idx[:, None]
      input_tokens = input_tokens[jnp.arange(bs)[:, None], pos_ids]
    else:
      input_tokens = input_tokens[:,:-1]
    
    return input_tokens, target_tokens, loss_mask
    
  def get_target_sequence(self, input_tokens, mask, target_tokens=None, task_mask=None,
                          loss_mask=None, segment_ids=None, cur_index=None, pos_ids=None):
    cfg = self.config
    vae_cfg = self.vae_config
    x = self.embedding_layer(input_tokens)

    if cur_index is not None:
      pos_emb = jax.lax.dynamic_slice(self.pos_emb_cache, (cur_index, 0), (1, self.pos_emb_cache.shape[-1]))[None, :, :]
    else:
      pos_emb = self.pos_emb_cache[:x.shape[1]][None,:,:]
    
    pos_emb = jnp.tile(pos_emb, (x.shape[0], 1, 1))

    if "rope" not in cfg.image_pos_emb:
      x += pos_emb      

    if "llama_rope" in cfg.image_pos_emb:
      modality_emb = param_with_axes(
        "modality_embedding",
        nn.initializers.normal(stddev=0.02),
        (cfg.emb_dim,),
        axes=(('embed',)),
      )
      x += modality_emb[None, None, :].astype(cfg.dtype)

    if mask is None:
      mask = jnp.ones((x.shape[0], x.shape[1]), jnp.int32)

    if cfg.dalle_attn_mask:
      attn_pattern_mask = jnp.tile(self.attn_mask[None,:,:,:], [x.shape[0], 1, 1, 1])
      attn_pattern_mask = jnp.array(attn_pattern_mask, cfg.dtype)
    else:
      # use full mask if we are not using dalle attn mask.
      attn_pattern_mask = jnp.tile(self.attn_mask[None,-1,:,:], [x.shape[0], 4, 1, 1])
      attn_pattern_mask = jnp.array(attn_pattern_mask, cfg.dtype)

    if cfg.dynamic_unk_mask and task_mask is not None:
      noise_mask = 1 - task_mask
      # shift the mask by 1
      noise_mask = jnp.concatenate([
        jnp.ones((noise_mask.shape[0], 1), dtype=cfg.dtype),
        noise_mask[:,:-1]], axis=1)

      dynamic_unk_mask = layers.make_attention_mask(noise_mask, noise_mask, dtype=cfg.dtype)
      identity_mask = jnp.identity(x.shape[1], dtype=cfg.dtype)
      dynamic_unk_mask = jnp.array(jnp.logical_or(dynamic_unk_mask, identity_mask), dtype=cfg.dtype)
      attn_pattern_mask = layers.combine_masks(dynamic_unk_mask, attn_pattern_mask, dtype=cfg.dtype)

    seq = TargetSequence(
      x, pos_emb, jnp.array(IMAGE_MODALITY_INDEX, dtype=jnp.int32), mask, attn_pattern_mask=attn_pattern_mask,
      subsegments=segment_ids, target_tokens=target_tokens, loss_mask=loss_mask)

    if pos_ids is not None:
      mat = seq_features.build_packing_matrix_from_pos(pos_ids, x.shape[1])
      seq = seq_features.multiply_seq_dim(seq, mat)

    return seq

  @nn.compact
  def __call__(self, image, mask=None, loss_mask=None, task_mask=None, init=False, segment_ids=None,
              decode=False, decode_length=None, cur_index=None, pos_ids=None):
    
    cfg = self.config
    if decode:
      return self.get_target_sequence(image, mask, segment_ids, cur_index=cur_index)
    else:
      input_tokens, target_tokens, loss_mask = self.target_image_to_seq(
          image, loss_mask, init, task_mask)

      return self.get_target_sequence(input_tokens, mask, target_tokens, task_mask,
                                      loss_mask, segment_ids, pos_ids=pos_ids)


@dataclass
class TargetImageDVAEEmbedder(ModalityEncoder):
  """Builds target tokens for a target image"""
  config: ImageViTVQGANConfig

  def preprocess_inputs(
      self, features: Dict, output_features, sequence_length) -> Optional[Dict[str, tf.Tensor]]:
    image_target_size = IMAGE_TARGET_SIZE
    image_target_d = IMAGE_TARGET_D
    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    image_targets = features.pop("image_targets", None)
    image_target_masks = features.pop("image_target_masks", None)
    image_target_task_masks = features.pop("image_target_task_masks", None)
    if image_targets is None:
      assert image_target_masks is None
      assert 'image_target_loss_masks' not in features
      assert image_target_task_masks is None
      image_targets = tf.zeros(image_target_size+[0], tf.float32)
      image_target_masks = tf.zeros([0], tf.int32)
      image_target_task_masks = tf.zeros([0], tf.int32)
    else:
      image_targets = image_targets * 2.0 - 1  # VAE pre-processing
      # In case the dimension were unknown
      image_targets = tf.ensure_shape(image_targets, image_target_size + [3])
      assert image_target_masks is not None
      if len(image_target_masks.shape) == 1:
        # Given mask is on the patches rather then pixels, used in depth_preprocessing
        image_target_masks = image_target_masks
      else:
        image_target_masks = tf.image.resize(
          tf.expand_dims(image_target_masks, -1),
          target_padding_size,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)
      if image_target_task_masks is None:
        image_target_task_masks = tf.zeros(image_target_masks.shape, tf.int32)
      else:
        if len(image_target_task_masks.shape) == 1:
          image_target_task_masks = image_target_task_masks
        else:
          image_target_task_masks = tf.image.resize(
            tf.expand_dims(image_target_task_masks, -1),
            target_padding_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_target_task_masks = tf.cast(tf.reshape(image_target_task_masks, [-1]), tf.int32)

    loss_mask = features.get('image_target_loss_masks', image_target_masks)

    return dict(
      image=image_targets,
      mask=image_target_masks,
      loss_mask=loss_mask,
      task_mask=image_target_task_masks,
    )

  def convert_inputs(self, features: Optional[Dict], sequence_length) -> Dict[str, tf.Tensor]:
    image_len = (IMAGE_TARGET_SIZE[0] // IMAGE_TARGET_D) \
        * (IMAGE_TARGET_SIZE[1] // IMAGE_TARGET_D)
    image_trm_len =  image_len
    image_shape = IMAGE_TARGET_SIZE + [3]
    if tf.shape(features["image"])[-1] == 0:
      features = {
        'image': tf.zeros(image_shape, tf.float32),
        'mask': tf.zeros((image_trm_len,), tf.int32),
        'loss_mask': tf.zeros((image_len,), tf.int32),
        'task_mask': tf.zeros((image_trm_len,), tf.int32),
      }
    features["loss_mask"] = tf.ensure_shape(features["loss_mask"], [image_len])
    features["mask"] = tf.ensure_shape(features["mask"], [image_len])
    features["task_mask"] = tf.ensure_shape(features["task_mask"], [image_len])

    features["image"] = tf.ensure_shape(features["image"], image_shape)
    return features

  def get_output_features(self) -> Dict[str, seqio.Feature]:
    out = {
      "image": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
      "mask": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
      "loss_mask": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
      "task_mask": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
    }

    return out

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return ImageViTVQGAN(config, self.config, shared_embedding)

  def get_decoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return BasicDecoder(config.image_vocab_size, config, shared_embedding)


class AudioViTVQGAN(nn.Module):
  config: T5Config
  vae_config: AudioViTVQGANConfig
  embedding_layer: nn.Module

  def setup(self):
    cfg = self.config
    vae_cfg = self.vae_config
    self.grid_size = [
        self.config.default_audio_size[0] // self.vae_config.patch_size[0],
        self.config.default_audio_size[1] // self.vae_config.patch_size[1],
    ]

    self.discrete_vae = ASTVQGAN(self.vae_config)
    assert cfg.emb_dim == self.embedding_layer.embedding.shape[-1]

    # construct the row, col and conv mask.
    row_mask = get_row_mask(self.grid_size[0], self.grid_size[1])
    col_mask = get_col_mask(self.grid_size[0], self.grid_size[1])
    conv_mask = get_conv_mask(self.grid_size[0], self.grid_size[1])
    full_mask = _init_mask(self.grid_size[0], self.grid_size[1])
    
    self.attn_mask = jnp.stack([
        jnp.array(row_mask, cfg.dtype), 
        jnp.array(col_mask, cfg.dtype),
        jnp.array(conv_mask, cfg.dtype), 
        jnp.array(full_mask, cfg.dtype),
    ])

    self.pos_emb_cache = layers.get_2d_position_embedding(
        cfg.audio_pos_emb, vae_cfg.default_input_size, vae_cfg.patch_size,
        cfg.emb_dim, cfg.head_dim, 3, cfg.dtype)

  def target_audio_to_seq(self, audio, loss_mask=None, init=False, 
                          task_mask=None):
    
    cfg = self.config
    bs = audio.shape[0]

    #  since the vit-vqgan is [128, 256] we need to tranpose this first.
    audio = jnp.transpose(audio, [0,2,1,3])

    if init: _ = self.discrete_vae(audio)
    target_tokens = self.discrete_vae.get_codebook_indices(audio)
    
    # transform the target back.
    target_tokens = jnp.reshape(target_tokens, [bs, self.grid_size[1], self.grid_size[0]])
    target_tokens = jnp.reshape(jnp.transpose(target_tokens, [0, 2, 1]), [bs, -1])
  
    # 0: start token
    # 1: [MASK] token
    # from 2: normal tokens
    target_tokens = target_tokens + 2
    target_tokens = jax.lax.stop_gradient(target_tokens)

    # task_mask:
    #   0 if we should keep the corresponding token
    #   1 if we should replace the corresponding token with the [MASK] token
    if task_mask is not None and cfg.unk_mask_token:
      input_tokens = (
        target_tokens * (1 - task_mask.astype(jnp.int32)) +
        jnp.ones(target_tokens.shape, dtype=jnp.int32) * task_mask.astype(jnp.int32)
      )
    else:
      input_tokens = target_tokens

    input_tokens = jnp.concatenate([
      jnp.zeros((input_tokens.shape[0], 1), dtype=jnp.int32),
      input_tokens], axis=1)

    if cfg.shift_back_input_token: 
      if task_mask is not None:
        start_idx = (task_mask.sum(axis=-1) > 0).astype(jnp.int32)
      else:
        start_idx = jnp.zeros((bs,), dtype=jnp.int32)
      pos_ids = jnp.arange(target_tokens.shape[1])[None, :] + start_idx[:, None]
      input_tokens = input_tokens[jnp.arange(bs)[:, None], pos_ids]
    else:
      input_tokens = input_tokens[:,:-1]

    return input_tokens, target_tokens, loss_mask

  def get_target_sequence(self, input_tokens, mask, target_tokens=None, task_mask=None,
                          loss_mask=None, segment_ids=None, cur_index=None):

    cfg = self.config

    x = self.embedding_layer(input_tokens)

    if cur_index is not None:
      pos_emb = jax.lax.dynamic_slice(self.pos_emb_cache, (cur_index, 0), (1, self.pos_emb_cache.shape[-1]))[None, :, :]
    else:
      pos_emb = self.pos_emb_cache[:x.shape[1]][None,:,:]
    
    pos_emb = jnp.tile(pos_emb, (x.shape[0], 1, 1))

    if "rope" not in cfg.audio_pos_emb:
      x += pos_emb      

    if "llama_rope" in cfg.audio_pos_emb:
      modality_emb = param_with_axes(
        "modality_embedding",
        nn.initializers.normal(stddev=0.02),
        (cfg.emb_dim,),
        axes=(('embed',)),
      )
      x += modality_emb[None, None, :].astype(cfg.dtype)

    if mask is None:
      mask = jnp.ones((x.shape[0], x.shape[1]), jnp.int32)
    
    if cfg.dalle_attn_mask:
      attn_pattern_mask = jnp.tile(self.attn_mask[None,:,:,:], [x.shape[0], 1, 1, 1])
      attn_pattern_mask = jnp.array(attn_pattern_mask, cfg.dtype)
    else:
      attn_pattern_mask = jnp.tile(self.attn_mask[None,-1,:,:], [x.shape[0], 4, 1, 1])
      attn_pattern_mask = jnp.array(attn_pattern_mask, cfg.dtype)

    if cfg.dynamic_unk_mask and task_mask is not None:
      noise_mask = 1 - task_mask
      
      noise_mask = jnp.concatenate([
        jnp.ones((noise_mask.shape[0], 1), dtype=cfg.dtype),
        noise_mask[:,:-1]], axis=1)
            
      dynamic_unk_mask = layers.make_attention_mask(noise_mask, noise_mask, dtype=cfg.dtype)
      identity_mask = jnp.identity(x.shape[1], dtype=cfg.dtype)
      dynamic_unk_mask = jnp.array(jnp.logical_or(dynamic_unk_mask, identity_mask), dtype=cfg.dtype)
      attn_pattern_mask = layers.combine_masks(dynamic_unk_mask, attn_pattern_mask, dtype=cfg.dtype)

    seq = TargetSequence(
      x, pos_emb, jnp.array(AUDIO_MODALITY_INDEX, dtype=jnp.int32), mask, attn_pattern_mask=attn_pattern_mask,
      subsegments=segment_ids, target_tokens=target_tokens, loss_mask=loss_mask)

    return seq

  @nn.compact
  def __call__(self, audio, mask=None, loss_mask=None, task_mask=None, init=False,  segment_ids=None,
              decode=False, decode_length=None, cur_index=None):
    if decode:
      return self.get_target_sequence(audio, mask, segment_ids, cur_index=cur_index)
    else:
      input_tokens, target_tokens, loss_mask = self.target_audio_to_seq(
          audio, loss_mask, init, task_mask)

      return self.get_target_sequence(
        input_tokens, mask, target_tokens, task_mask, loss_mask, segment_ids)


@dataclass
class TargetAudioDVAEEmbedder(ModalityEncoder):
  """Builds target tokens for a target audio segment"""
  config: AudioViTVQGANConfig

  def preprocess_inputs(
      self, features: Dict, output_features, sequence_length) -> Optional[Dict[str, tf.Tensor]]:
    
    target_size = AUDIO_TARGET_SIZE
    target_d = AUDIO_TARGET_D

    target_padding_size = tf.constant(
      np.array(target_size) / target_d, tf.int32)

    targets = features.pop("audio_targets", None)
    target_masks = features.pop("audio_target_masks", None)
    target_task_masks = features.pop("audio_target_task_masks", None)

    if targets is None:
      assert target_masks is None
      assert 'audio_target_loss_masks' not in features
      assert target_task_masks is None
      targets = tf.zeros(target_size+[0], tf.float32)
      target_masks = tf.zeros([0], tf.int32)
      target_task_masks = tf.zeros([0], tf.int32)
    else:
      targets = (targets - AUDIOSET_MEAN) / AUDIOSET_STD
      # In case the dimension were unknown
      targets = tf.ensure_shape(targets, target_size + [1])
      assert target_masks is not None
      if len(target_masks.shape) == 1:
        # Given mask is on the patches rather then pixels, used in depth_preprocessing
        target_masks = target_masks
      else:
        target_masks = tf.image.resize(
          tf.expand_dims(target_masks, -1),
          target_padding_size,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      target_masks = tf.cast(tf.reshape(target_masks, [-1]), tf.int32)
      if target_task_masks is None:
        target_task_masks = tf.zeros(target_masks.shape, tf.int32)
      else:
        if len(target_task_masks.shape) == 1:
          target_task_masks = target_task_masks
        else:
          target_task_masks = tf.image.resize(
            tf.expand_dims(target_task_masks, -1),
            target_padding_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
          target_task_masks = tf.cast(tf.reshape(target_task_masks, [-1]), tf.int32)

    loss_mask = features.get('audio_target_loss_masks', target_masks)

    return dict(
      audio=targets,
      mask=target_masks,
      loss_mask=loss_mask,
      task_mask=target_task_masks,
    )

  def convert_inputs(self, features: Optional[Dict], sequence_length) -> Dict[str, tf.Tensor]:
    
    target_len = (AUDIO_TARGET_SIZE[0] // AUDIO_TARGET_D) * (AUDIO_TARGET_SIZE[1] // AUDIO_TARGET_D)
    target_shape = AUDIO_TARGET_SIZE + [1]
    target_trm_len =  target_len
    
    if features is None or tf.shape(features["audio"])[-1] == 0:
      # Replace dummy features with full-sized masked features to keep shape consistent
      target = tf.zeros(target_shape, tf.float32)
      features = {
        'audio': target,
        'mask': tf.zeros((target_trm_len,), tf.int32),
        'loss_mask': tf.zeros((target_len,), tf.int32),
        'task_mask': tf.zeros((target_trm_len,), tf.int32),
      }

    # If statement can screw up shape info, fix here:
    features["mask"] = tf.ensure_shape(features["mask"], [target_trm_len])
    features["loss_mask"] = tf.ensure_shape(features["loss_mask"], [target_len])
    features["task_mask"] = tf.ensure_shape(features["task_mask"], [target_trm_len])

    features["audio"] = tf.ensure_shape(features["audio"], target_shape)
    return features

  def get_output_features(self) -> Dict[str, seqio.Feature]:
    out = {
      "audio": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
      "mask": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
      "loss_mask": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
      "task_mask": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
    }
    return out

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return AudioViTVQGAN(config, self.config, shared_embedding)

  def get_decoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return BasicDecoder(config.audio_vocab_size, config, shared_embedding)
