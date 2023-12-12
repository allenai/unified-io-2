"""UIO 2 Encoder/Decoder Transformer, based on the T5.1.1 Transformer model."""
import logging
from typing import Callable, Iterable, Optional, List

import jax

from t5x.examples.unified_io import modality_processing
from t5x.examples.unified_io.config import T5Config

from flax import linen as nn, traverse_util
import jax.numpy as jnp

from t5x.examples.unified_io import layers
from t5x.examples.unified_io.seq_features import InputSequence
from t5x.examples.unified_io import seq_features
from flax.linen.partitioning import ScanIn, scan_with_axes
import clu.metrics as clu_metrics


PyTreeDef = type(jax.tree_util.tree_structure(None))
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: T5Config

  @nn.compact
  def __call__(self, inputs, encoder_mask=None, abs_bias=None, sinusoids=None, deterministic=False):
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_attention_layer_norm')(
            inputs)

    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        qk_norm=cfg.qk_norm,
        name='attention')(
            x, x, encoder_mask, None, abs_bias=abs_bias,
            q_sinusoids=sinusoids, k_sinusoids=sinusoids, deterministic=deterministic)

    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)

    x = x + inputs

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
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

    if cfg.scan_layers:
      return y, None
    else:
      return y


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: T5Config

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               decoder_bias=None,
               cross_abs_pos_bias=None,
               decoder_sinusoids=None,
               encoder_sinusoids=None, 
               attn_pattern_mask=None,
               enable_xattention=True,
               ):

    cfg = self.config

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)

    # Self-attention block
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        qk_norm=cfg.qk_norm,
        name='self_attention')(
            x,
            x,
            decoder_mask,
            decoder_bias,
            q_sinusoids=decoder_sinusoids,
            k_sinusoids=decoder_sinusoids,
            deterministic=deterministic,
            attn_pattern_mask=attn_pattern_mask,
            decode=decode)

    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)

    x = x + inputs
    
    if enable_xattention:
      # Encoder-Decoder block.
      y = layers.LayerNorm(
          dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
              x)

      y = layers.MultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          head_dim=cfg.head_dim,
          dropout_rate=cfg.dropout_rate,
          float32_logits=cfg.float32_attention_logits,
          qk_norm=cfg.qk_norm,
          name='encoder_decoder_attention')(
              y,
              encoded,
              encoder_decoder_mask, 
              cross_abs_pos_bias,
              q_sinusoids=decoder_sinusoids,
              k_sinusoids=encoder_sinusoids,
              deterministic=deterministic)

      y = nn.Dropout(
          rate=cfg.dropout_rate, broadcast_dims=(-2,))(
              y, deterministic=deterministic)
    
      y = y + x
    else:
      y = x

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
    z = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + y

    if self.config.scan_layers:
      return z, None
    else:
      return z


class Encoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config

  @nn.compact
  def __call__(self, seq: InputSequence, deterministic=False):
    cfg = self.config
    embed = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      seq.embed, deterministic=deterministic)
    embed = embed.astype(cfg.dtype)

    mask = layers.make_attention_mask(seq.mask, seq.mask, dtype=cfg.dtype)
    if seq.segment_ids is not None:
      # Only attend between items belonging to the same segment
      mask = mask * jnp.expand_dims(seq.segment_ids[:, :, None] == seq.segment_ids[:, None, :], 1)
    pos_emb = seq.position_embed
    sinusoids = pos_emb if (pos_emb is not None and pos_emb.shape[-1] != embed.shape[-1]) else None

    if cfg.scan_layers:
      scan_axis = cfg.scan_axis
      initializing = self.is_mutable_collection('params')
      params_spec = (scan_axis if initializing else ScanIn(scan_axis))
      cache_spec = 0

      embed, _ = scan_with_axes(
        EncoderLayer,
        variable_axes={
          'params': params_spec,
          'cache': cache_spec,
        },
        split_rngs={
          'params': True,
          'dropout': True
        },
        in_axes=(nn.broadcast,)*5,
        length=cfg.num_encoder_layers,
        unroll=cfg.num_encoder_layers if cfg.scan_unroll == "all" else cfg.scan_unroll,
        axis_name='layers')(
        config=cfg, name='encoder')(embed, mask, None, sinusoids, deterministic)
    else:
      for lyr in range(cfg.num_encoder_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        embed = EncoderLayer(config=cfg, name=f'layers_{lyr}')(
          embed, mask, None, sinusoids, deterministic)

    embed = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(embed)
    embed = nn.Dropout(rate=cfg.dropout_rate)(embed, deterministic=deterministic)
    return embed


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: T5Config

  @nn.compact
  def __call__(self,
               encoded,
               decoder_embedding,
               decoder_pos_emb=None,
               decoder_attn_mask=None,
               encoder_pos_emb=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               decoder_bias=None,
               attn_pattern_mask=None,
               cur_index=None):

    cfg = self.config
    assert decoder_embedding.ndim == 3  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = decoder_embedding
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    cross_abs_pos_bias = None
    use_rope = (
      encoder_pos_emb is not None and decoder_pos_emb is not None and 
      decoder_embedding.shape[-1] != decoder_pos_emb.shape[-1] and
      decoder_pos_emb.shape[-1] == encoder_pos_emb.shape[-1]
    )
    encoder_sinusoids = encoder_pos_emb if use_rope else None
    decoder_sinusoids = decoder_pos_emb if use_rope else None

    if cfg.scan_layers:
      assert attn_pattern_mask is None
      initializing = self.is_mutable_collection('params')
      params_spec = (
        cfg.scan_axis if initializing else ScanIn(cfg.scan_axis))
      cache_spec = 0

      y, _ = scan_with_axes(
        DecoderLayer,
        variable_axes={
          'params': params_spec,
          'cache': cache_spec
        },
        split_rngs={
          'params': True,
          'dropout': True
        },
        in_axes=(nn.broadcast,)*9,
        length=cfg.num_decoder_layers,
        unroll=cfg.num_decoder_layers if cfg.scan_unroll == "all" else cfg.scan_unroll,
        axis_name='layers')(
        config=cfg,
        name='decoder')(
          y, encoded, decoder_attn_mask, encoder_decoder_mask,
          deterministic, decode, decoder_bias, cross_abs_pos_bias,
          decoder_sinusoids, encoder_sinusoids,
      )
    else:
      for lyr in range(cfg.num_decoder_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        
        if attn_pattern_mask is not None:
          if lyr == cfg.num_decoder_layers - 1:
            attn_pattern_lyr = attn_pattern_mask[:,2:3]
          elif (lyr - 1) % 4 == 0:
            attn_pattern_lyr = attn_pattern_mask[:,1:2]
          else:
            attn_pattern_lyr = attn_pattern_mask[:,0:1]
        else:
          attn_pattern_lyr = None

        enable_xattention = False
        if lyr % cfg.decoder_xattention_internval == 0 or lyr == (cfg.num_decoder_layers-1):
          enable_xattention = True
        
        y = DecoderLayer(
            config=cfg,
            name=f'layers_{lyr}')(
                y,
                encoded,
                decoder_mask=decoder_attn_mask,
                encoder_decoder_mask=encoder_decoder_mask,
                deterministic=deterministic,
                decode=decode,
                decoder_bias=decoder_bias,
                cross_abs_pos_bias=cross_abs_pos_bias,
                decoder_sinusoids=decoder_sinusoids,
                encoder_sinusoids=encoder_sinusoids,
                attn_pattern_mask=attn_pattern_lyr,
                enable_xattention=enable_xattention)

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    return y


class Transformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: T5Config

  def setup(self):
    cfg = self.config

    self.shared_text_embedding = layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='text_token_embedder')

    # depends on different method, we use different embedder here.
    self.shared_image_embedding = layers.Embed(
        num_embeddings=cfg.image_vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='image_token_embedder')

    self.shared_audio_embedding = layers.Embed(
        num_embeddings=cfg.audio_vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='audio_token_embedder')

    input_shared_embedding = {
      'text': self.shared_text_embedding,
    }

    target_shared_embedding = {
      'text': self.shared_text_embedding,
      'image': self.shared_image_embedding,
      'audio': self.shared_audio_embedding,
    }

    # Modality processing modules
    self.input_encoders = {k: v.get_encoder(self.config, input_shared_embedding.get(k, None))
                           for k, v in modality_processing.get_input_modalities().items()}

    self.target_encoders = {k: v.get_encoder(self.config, target_shared_embedding.get(k, None))
                           for k, v in modality_processing.get_target_modalities().items()}

    self.target_decoders = {k: v.get_decoder(self.config, target_shared_embedding.get(k, None))
                           for k, v in modality_processing.get_target_modalities().items()}
    
    # The transformers
    self.encoder = Encoder(config=cfg)
    self.decoder = Decoder(config=cfg)

  def decode_image_code(self, code_b):
    return self.target_encoders["image"].discrete_vae.decode_code(code_b)

  def decode_audio_code(self, code_b):
    return self.target_encoders["audio"].discrete_vae.decode_code(code_b)

  def get_audio_code(self, x):
    return self.target_encoders["audio"].discrete_vae.get_codebook_indices(x)

  def get_image_code(self, x):
    return self.target_encoders["image"].discrete_vae.get_codebook_indices(x)

  def sample(
      self, 
      encoded,
      encoder_masks,
      decoder_inputs,
      decoder_masks=None,
      enable_dropout=True,
      decode=False,
      cur_index=None,
      decode_length=None, 
      modality=None):
    cfg = self.config

    encoded, encoder_pos_emb = encoded
    target_seq = self.target_encoders[modality](
        decoder_inputs, decoder_masks, decode=True, 
        decode_length=decode_length, cur_index=cur_index)

    encoder_decoder_mask = layers.make_attention_mask(
        jnp.ones(target_seq.input_embedding.shape[:2]),
        encoder_masks,
        dtype=cfg.dtype)

    if decoder_masks is not None:
      decoder_attn_mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_masks,
        dtype=cfg.dtype)
    else:
      decoder_attn_mask = None

    hidden_state = self.decoder(
        encoded,
        decoder_pos_emb=target_seq.position_id,
        decoder_embedding=target_seq.input_embedding,
        decoder_attn_mask=decoder_attn_mask,
        encoder_pos_emb=encoder_pos_emb,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        decoder_bias=None,
        attn_pattern_mask = target_seq.attn_pattern_mask,
        cur_index=cur_index,
    )
    # given hidden state, forward to decoder.
    logits = self.target_decoders[modality](hidden_state, decode=True)

    return logits

  def decode(self, encoded_inputs, encoded_mask, target_features, horizontally_pack_targets=False):
    """
    Compute a forward pass using the given `encoded_inputs` and `encoded_mask` instead of
    "raw" input modalities, used to efficiently compute log-probabilities of multiple
    answer options for one input query
    """
    cfg = self.config
    target_parts = []
    target_features = traverse_util.unflatten_dict(target_features, sep="/")["targets"]

    for k, v in self.target_encoders.items():
      if target_features.get(k) is not None:
        target_parts.append(v(**target_features[k]))

    if any(x.subsegments is not None for x in target_parts):
      for x in target_parts:
        x.subsegments = jnp.ones((x.batch_size, x.seq_len), dtype=jnp.int32)

    target_tokens = [k.target_tokens for k in target_parts]
    loss_masks = [k.loss_mask for k in target_parts]
    for part in target_parts:
      part.loss_mask = None
      part.target_tokens = None

    if horizontally_pack_targets:
      target_seq = seq_features.pack_horizontally(target_parts, horizontally_pack_targets)
    else:
      target_seq = seq_features.concat_sequences(target_parts)

    encoder_decoder_mask = layers.make_attention_mask(target_seq.mask, encoded_mask, dtype=cfg.dtype)
    all_subsegments = target_seq.get_all_subsegments()

    decoder_attn_mask = layers.make_decoder_mask(
      decoder_target_tokens=target_seq.mask,
      dtype=cfg.dtype,
      decoder_segment_ids=all_subsegments)

    if target_seq.segment_ids is not None:
      raise NotImplementedError()

    hidden_state = self.decoder(
      encoded_inputs[0],
      decoder_pos_emb=target_seq.position_id,
      decoder_embedding=target_seq.input_embedding,
      decoder_attn_mask=decoder_attn_mask,
      encoder_pos_emb=encoded_inputs[1],
      encoder_decoder_mask=encoder_decoder_mask,
      deterministic=True,
      decode=False,
      decoder_bias=None,
      attn_pattern_mask=target_seq.attn_pattern_mask,
    )

    if not horizontally_pack_targets:
      embedding_parts = seq_features.split_sequence_dim(
        hidden_state, [x.seq_len for x in target_parts])
    else:
      embedding_parts = seq_features.split_and_unpack(
        hidden_state, [x.mask for x in target_parts])

    # Embedding parts are a list of per-modality hidden states of the same sequence
    # length as the input target sequences
    assert len(embedding_parts) == len(target_parts)
    assert all(a.shape[1] == b.shape[1] for a, b in zip(embedding_parts, target_tokens))

    logits = {}
    for (name, decoder), state, targets, mask in zip(
        self.target_decoders.items(), embedding_parts, target_tokens, loss_masks):
      logits[name] = (decoder(state), targets, mask)

    return logits

  def encode(self, features, enable_dropout=True, horizontally_pack_inputs=None):
    features = traverse_util.unflatten_dict(features, sep="/")["inputs"]
    input_parts = []
    for k, v in self.input_encoders.items():
      inputs = {'enable_dropout': enable_dropout, 'use_constraints': True, **features[k]}
      input_parts.append(v(**inputs))
    if horizontally_pack_inputs:
      seq = seq_features.pack_horizontally(input_parts, horizontally_pack_inputs)
    else:
      seq = seq_features.concat_sequences(input_parts)
    embed = self.encoder(seq, not enable_dropout)
    return (embed, seq.position_embed), seq.mask

  @nn.compact
  def __call__(
    self,
    features,
    enable_dropout: bool = True,
    horizontally_pack_inputs: Optional[int]=None,
    horizontally_pack_targets: Optional[int]=None,
    fold=False, init=False, decode=False, use_constraints=True,
    return_packing_stats=False
  ):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      features: input data to the encoder.
      enable_dropout: Ensables dropout if set to True.
      horizontally_pack_inputs: Pack inputs horizontally to this max length
      horizontally_pack_targets: Pack targets horizontally to this max length
      fold: Fold examples to half batch size and 2x sequence length, used for packing
      decode: Whether to prepare and use an autoregressive cache.
      init: Whether we are initializing, if True all parameters will used to they
            they will be correctly initialized by flax
      use_constraints: Turn on constrains for input modalities, should only be turned on
                       if training and batches are changed to conform to constraints
      return_packing_stats: Return statistics about masking and packing
    Returns:
      modality_logits: a dictionary for each target modality, the values are tuples with:
        logits: token probabilities for each target token
        target_ids: target tokens
        mask: identifying which target tokens are valid
    """
    if fold is True:
      fold = 2
    cfg = self.config

    # Build the inut sequence features
    features = traverse_util.unflatten_dict(features, sep="/")
    input_features = features["inputs"]
    input_parts: List[InputSequence] = []
    for k, v in self.input_encoders.items():
      inputs = {'enable_dropout': enable_dropout, 'use_constraints': use_constraints,
                **input_features[k]}
      input_parts.append(v(**inputs))

    if fold:
      input_parts_to_pack = [seq_features.fold_sequence(x, n=fold) for x in input_parts]
    else:
      input_parts_to_pack = input_parts
    n_input_tokens = jnp.stack([jnp.sum(x.mask, -1) for x in input_parts_to_pack], -1).sum(-1)

    if horizontally_pack_inputs:
      input_seq = seq_features.pack_horizontally(input_parts_to_pack, horizontally_pack_inputs)
    else:
      input_seq = seq_features.concat_sequences(input_parts_to_pack)
    n_packed_input_tokens = input_seq.mask.sum(-1)

    logging.info(f"Input parts=({input_parts[0].batch_size}, {[x.seq_len for x in input_parts]}), "
                 f"final length=({input_seq.batch_size}, {input_seq.seq_len})")

    # Do the encoding
    embed = self.encoder(input_seq, deterministic=not enable_dropout)

    target_parts = []
    target_features = features["targets"]

    for k, v in self.target_encoders.items():
      if target_features.get(k) is not None:
        target_parts.append(v(**target_features[k], init=init))
    
    target_tokens = [k.target_tokens for k in target_parts]
    loss_masks = [k.loss_mask for k in target_parts]
    for part in target_parts:
      part.loss_mask = None
      part.target_tokens = None

    has_any_target = jnp.stack([jnp.sum(x.mask, -1) for x in target_parts], -1).sum(-1) > 0
    if fold:
      target_parts = [seq_features.fold_sequence(x, n=fold) for x in target_parts]
    n_target_tokens = jnp.stack([jnp.sum(x.mask, -1) for x in target_parts], -1).sum(-1)

    if horizontally_pack_targets:
      target_seq = seq_features.pack_horizontally(target_parts, horizontally_pack_targets)
    else:
      target_seq = seq_features.concat_sequences(target_parts)
    n_packed_target_tokens = jnp.stack([jnp.sum(x.mask, -1) for x in target_parts], -1).sum(-1)

    logging.info(f"Target parts=({target_parts[0].batch_size}, {[x.seq_len for x in target_parts]}), "
                 f"final length=({target_seq.batch_size}, {target_seq.seq_len})")

    # Build the decoder masks TODO move into the decoder?
    encoder_decoder_mask = layers.make_attention_mask(target_seq.mask, input_seq.mask, dtype=cfg.dtype)
    all_subsegments = target_seq.get_all_subsegments()

    decoder_attn_mask = layers.make_decoder_mask(
      decoder_target_tokens=target_seq.mask,
      dtype=cfg.dtype,
      decoder_segment_ids=all_subsegments)

    if target_seq.segment_ids is not None:
      cross_seg_mask = jnp.expand_dims(target_seq.segment_ids, -1) == jnp.expand_dims(input_seq.segment_ids, -2)
      encoder_decoder_mask = encoder_decoder_mask * jnp.expand_dims(cross_seg_mask, 1)

    # Do the decoding and output the feature vector for transformers.
    hidden_state = self.decoder(
      embed,
      decoder_pos_emb=target_seq.position_id,
      decoder_embedding=target_seq.input_embedding,
      decoder_attn_mask=decoder_attn_mask,
      encoder_pos_emb=input_seq.position_embed,
      encoder_decoder_mask=encoder_decoder_mask,
      deterministic=not enable_dropout,
      decode=decode,
      decoder_bias=None, 
      attn_pattern_mask=target_seq.attn_pattern_mask,
    )

    if not horizontally_pack_targets:
      embedding_parts = seq_features.split_sequence_dim(
        hidden_state, [x.seq_len for x in target_parts])
    else:
      embedding_parts = seq_features.split_and_unpack(
        hidden_state, [x.mask for x in target_parts])

    if fold:
      bs, _, dim = embedding_parts[0].shape
      embedding_parts = [jnp.reshape(x, [bs*fold, x.shape[1]//fold, dim]) for x in embedding_parts]

    # Embedding parts are a list of per-modality hidden states of the same sequence
    # length as the input target sequences
    assert len(embedding_parts) == len(target_parts)
    assert all(a.shape[1] == b.shape[1] for a, b in zip(embedding_parts, target_tokens))

    modality_logits = {}
    for (name, decoder), state, targets, mask in zip(
        self.target_decoders.items(), embedding_parts, target_tokens, loss_masks):
      modality_logits[name] = (decoder(state), targets, mask)

    if not return_packing_stats:
      return modality_logits
    else:
      batch_size = input_seq.batch_size
      target_overflow = (n_target_tokens != n_packed_target_tokens).sum()
      input_overflow = (n_input_tokens != n_packed_input_tokens).sum()
      metrics ={
        "packing/target_nonpadding_fraction": clu_metrics.Average(
          total=n_target_tokens.sum() / target_seq.seq_len, count=batch_size),
        "packing/input_nonpadding_fraction": clu_metrics.Average(
          total=n_input_tokens.sum() / input_seq.seq_len, count=batch_size),
        "packing/input_overflow": clu_metrics.Average(
          total=input_overflow, count=batch_size),
        "packing/target_overflow": clu_metrics.Average(
            total=target_overflow, count=batch_size),
        "packing/n-valid": clu_metrics.Average(
          total=has_any_target.sum(), count=has_any_target.shape[0])
      }
      if "image_history" in self.input_encoders:
        # Useful for making sure constraints are working correctly
        mask = features["inputs"]["image_history"]["mask"]
        metrics["packing/image_history_per_example"] = clu_metrics.Average(jnp.any(mask, - 1).sum(), mask.shape[0])
      if "audio_history" in self.input_encoders:
        mask = features["inputs"]["audio_history"]["mask"]
        metrics["packing/audio_history_per_example"] = clu_metrics.Average(jnp.any(mask, - 1).sum(), mask.shape[0])
      return modality_logits, metrics

