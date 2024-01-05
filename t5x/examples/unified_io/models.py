import functools
from typing import Callable, Mapping, MutableMapping, Tuple

import clu.metrics as clu_metrics
import gin
import jax
import numpy as np
import seqio
import tensorflow as tf
from absl import logging
from flax import linen as nn
from flax import traverse_util
from flax.core import scope as flax_scope
from flax.training import common_utils

from t5x import metrics as metrics_lib
from t5x import optimizers
from t5x.examples.unified_io import decoding
from t5x.examples.unified_io.aux_fns import clf_free_logit_mask_fn, clf_free_next_token_callback
from t5x.examples.unified_io.config import *
from t5x.examples.unified_io.modality_processing import UnifiedIOFeatureConverter
from t5x.losses import cross_entropy_with_logits
from t5x.models import DecodeFnCallable, BaseModel

Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
MetricsMap = metrics_lib.MetricsMap
PyTreeDef = type(jax.tree_util.tree_structure(None))


# Sentinel used instead of None to indicate missing values. For backward
# compatibility purposes; will be removed in an upcoming revision.
_NoValueSentinel = object()


class EncoderDecoderModel(BaseModel):
  """High level UnifiedIO 2 model interface

  The actual logits are computed by `module`, this provides loss and inference methods
  """

  FEATURE_CONVERTER_CLS = UnifiedIOFeatureConverter

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDef,
      decode_fn: DecodeFnCallable = decoding.beam_search,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
      loss_normalizing_by_weight_sum: Optional[bool] = True,
  ):
    if feature_converter_cls is not None:
      self.FEATURE_CONVERTER_CLS = feature_converter_cls  # pylint: disable=invalid-name
    self.module = module
    self._input_vocabulary = input_vocabulary
    self._output_vocabulary = output_vocabulary
    self._decode_fn = decode_fn
    self._label_smoothing = label_smoothing
    self._z_loss = z_loss
    self._loss_normalizing_factor = loss_normalizing_factor
    self._loss_normalizing_by_weight_sum = loss_normalizing_by_weight_sum
    super().__init__(optimizer_def=optimizer_def)

  @property
  def input_vocabulary(self):
    return self._input_vocabulary

  @property
  def output_vocabulary(self):
    return self._output_vocabulary

  @property
  def decode_fn(self):
    return self._decode_fn

  def loss_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
      label_smoothing: Optional[float] = None,
      z_loss: Optional[float] = None,
      loss_normalizing_factor: Union[Optional[float], object] = _NoValueSentinel,
      loss_normalizing_by_weight_sum: Union[Optional[float], object] = _NoValueSentinel,
      module_args=None, return_logits=False
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    # Default these to the constructor values. In the future, they may be
    # removed as parameters for `loss_fn`.
    label_smoothing = (
      self._label_smoothing if label_smoothing is None else label_smoothing)
    z_loss_param = self._z_loss if z_loss is None else z_loss
    if loss_normalizing_factor is _NoValueSentinel:
      loss_normalizing_factor = self._loss_normalizing_factor

    if loss_normalizing_by_weight_sum is _NoValueSentinel:
      loss_normalizing_by_weight_sum = self._loss_normalizing_by_weight_sum

    if module_args is None:
      module_args = {}
    module_args["return_packing_stats"] = True
    logits, stats = self._compute_logits(
      params, batch, dropout_rng, module_args=module_args)

    loss, metrics = multi_modality_loss(
      logits,
      z_loss_param, label_smoothing, loss_normalizing_factor,
      loss_normalizing_by_weight_sum
    )
    metrics.update(stats)
    if return_logits:
      return loss, metrics, logits
    else:
      return loss, metrics

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    if input_types is None:
      raise NotImplementedError()
    initial_variables = self.module.init(
      rng,
      {k: jnp.ones(input_shapes[k], input_types[k]) for k in input_shapes},
      enable_dropout=False, init=True
    )
    return initial_variables

  def _compute_logits(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None,
      mutable: flax_scope.CollectionFilter = False,
      module_args=None,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng, 'drop_path': dropout_rng, 'layer_drop': dropout_rng} if dropout_rng is not None else None
    if module_args is None:
      module_args = {}
      
    return self.module.apply(
      {'params': params},
      batch,
      decode=False, enable_dropout=rngs is not None,
      rngs=rngs, mutable=mutable, **module_args
    )

  def _compute_logits_from_slice(
      self, flat_ids: jnp.ndarray, flat_cache: Mapping[str, jnp.ndarray], cur_index: int,
      live_seqs: jnp.ndarray, params: PyTreeDef, encoded_inputs: jnp.ndarray, encoder_masks: jnp.ndarray,
      length: int, modality: str, post_process_logit_fn=None) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""

    flat_logits, new_vars = self.module.apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        encoder_masks,  # only needed for encoder padding mask
        flat_ids,
        enable_dropout=False,
        decode=True,
        decode_length=length,
        cur_index=cur_index,
        mutable=['cache'],
        modality=modality,
        method=self.module.sample)

    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']

    if post_process_logit_fn is not None:
      flat_logits = post_process_logit_fn(flat_logits, cur_index)

    return flat_logits, new_flat_cache

  def predict_batch_with_aux(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      length=64,
      modality="text",
      decode_rng: Any = None,
      top_k: int = 0,
      top_p: float = 1.0,
      temperature: float = 1.0,
      alpha: float = 0,
      repetition_penalty: float = None,
      horizontally_pack_inputs=None,
      negative_prompt=None,
      logit_mask_fn=None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Generate a prediction, modified from t5x.model.EncoderDecoderModel.predict_batch_with_aux

      params: model parameters.
      batch: a batch of inputs.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      length: int, max number of output tokens
      modality: str, text, image, or audio, output modality to generate
      decode_rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      top_k: Top k sampling to use with temperature sampling
      top_p: Top p sampling to use with temperature sampling
      temperature: Temperature to use with temperature sampling
      alpha: Alpha to use for classifier free guidance, ignored if `negative_prompt` is None
      repetition_penalty: repetition penalty
      horizontally_pack_inputs: Pack encoder inputs to length `horizontally_pack_inputs`, increases
                                performance but assumes there are at most `horizontally_pack_inputs`
                                non-masked tokens in the inputs
      negative_prompt: input batch to use for classifier free sampling, same shapes as `batch`
      logit_mask_fn: Function to apply to logits each step
    """
    if "choices" in batch:
      assert modality == "text"
      logging.info("Answer options found, using max-probability selection")
      return self.predict_with_answer_options(params, batch)

    # [batch, input_len]
    text_encoder_inputs = batch['inputs/text/tokens']

    text_type = text_encoder_inputs.dtype
    bs = text_encoder_inputs.shape[0]

    # Build dummy target features that reflect the target length/modality
    # These are used to initialize the cache
    target_features = {}
    if modality == "image":
      target_image = jnp.ones((bs, 256, 256, 3), jnp.float32)
      target_features["image"] = dict(image=target_image, mask=None, loss_mask=None)

    if modality == "audio":
      target_audio = jnp.ones((bs, 256, 128, 1), jnp.float32)
      target_features["audio"] = dict(audio=target_audio, mask=None, loss_mask=None)

    text_len = length if modality == "text" else 1
    target_features["text"] = dict(
      inputs=jnp.ones((bs, text_len), text_type),
      targets=jnp.ones((bs, text_len), text_type),
      mask=None, pos_ids=None, segment_ids=None)

    target_features = traverse_util.flatten_dict(
      dict(targets=target_features), keep_empty_nodes=True, sep="/")
    batch = {k: v for k, v in batch.items() if k.startswith("inputs/")}
    batch.update(target_features)

    if negative_prompt is not None:
      bs = batch['inputs/text/tokens'].shape[0]
      np_bs = negative_prompt['inputs/text/tokens'].shape
      if np_bs == 1 and bs > 1:
        # Assume the negative prompt is the same of all inputs
        clf_free_batch = {k: jnp.array(np.repeat(v, bs, axis=0)) for k, v in negative_prompt.items()}
      elif np_bs == bs:
        # Assume give a unique prompt for each example
        clf_free_batch = {k: jnp.array(v) for k, v in negative_prompt.items()}
      else:
        raise ValueError(f"Negative prompt has batch {np_bs}, but input has batch {bs}")
      assert clf_free_batch['inputs/text/tokens'].shape[0] == bs

      # Makes sure the batch uses the same dummy target features as `batch`
      clf_free_batch.update(target_features)

      # Concat with the regular batch
      batch = {k: None if v is None else jnp.concatenate([v, clf_free_batch[k]])
               for k, v in batch.items()}
      bs = batch['inputs/text/tokens'].shape[0]
      eff_bs = bs // 2
    else:
      eff_bs = bs

    # Encode the inputs
    batch = traverse_util.unflatten_dict(batch, sep="/")
    encoded_inputs, encoder_masks = self.module.apply(
      {'params': params}, batch, enable_dropout=False,
      horizontally_pack_inputs=horizontally_pack_inputs,
      method=self.module.encode)

    encoded, encoder_pos_embedding = encoded_inputs
    encoded = decoding.flat_batch_beam_expand(encoded, num_decodes)
    encoder_pos_embedding = decoding.flat_batch_beam_expand(encoder_pos_embedding, num_decodes)
    encoder_masks = decoding.flat_batch_beam_expand(encoder_masks, num_decodes)
    encoded_inputs = (encoded, encoder_pos_embedding)

    # Set up the cache
    _, variables_with_cache = self.module.apply(
        {'params': params},
        batch,
        horizontally_pack_inputs=horizontally_pack_inputs,
        decode=True,
        enable_dropout=False,
        mutable=['cache']
    )
    cache = variables_with_cache['cache']

    post_process_logit_fn = logit_mask_fn
    next_token_callback = None
    if negative_prompt is not None:
      # Callbacks to run CLF free guidance
      assert logit_mask_fn is None
      post_process_logit_fn = functools.partial(
        clf_free_logit_mask_fn, alpha=alpha, num_decodes=num_decodes)
      next_token_callback = functools.partial(
        clf_free_next_token_callback, num_decodes=num_decodes)

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        post_process_logit_fn=post_process_logit_fn,
        encoded_inputs=encoded_inputs,
        encoder_masks=encoder_masks,
        length=length,
        modality=modality)

    if decoder_params is None:
      decoder_params = {}

    # For beam search, `decoder_prompt_inputs` is only used to obtain batch size
    # and max decode length information. For temperature sampling,
    # `decod_prompt_inputs` will be filled with the sampled ids.
    decoder_prompt_inputs = jnp.zeros([bs, length], text_type)

    scanned = hasattr(self.module.config, 'scan_layers') and self.module.config.scan_layers

    logging.info(f"Decoding {modality}, len={length} fn={self._decode_fn}, "
                 f"logit_mask_fn={logit_mask_fn}, np={negative_prompt} num_decodes={num_decodes}, "
                 f"topk={top_k}, topp={top_p}, temperature={temperature}")

    if self._decode_fn == decoding.beam_search:
      if negative_prompt is not None or next_token_callback is not None:
        raise NotImplementedError()
      decodes, scores, logprobs = self._decode_fn(
          inputs=decoder_prompt_inputs,
          cache=cache,
          alpha=0.0,
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=self.output_vocabulary.eos_id,
          num_decodes=num_decodes,
          cache_offset=1 if scanned else 0,
          repetition_penalty=repetition_penalty,
          **decoder_params)
    else:
      if repetition_penalty:
        raise NotImplementedError()
      # TODO support logprobs in non-beam search decoding
      decodes, scores, logprobs = self._decode_fn(
          inputs=decoder_prompt_inputs,
          cache=cache,
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=self.output_vocabulary.eos_id,
          num_decodes=num_decodes,
          topk=top_k,
          topp=top_p,
          temperature=temperature,
          cache_offset=1 if scanned else 0,
          decode_rng = decode_rng,
          next_token_callback=next_token_callback,
          **decoder_params)

    out = {
      f"{modality}-scores": scores[:eff_bs],
      f"{modality}-logprobs": logprobs[:eff_bs]
    }

    if modality == 'text':
      if return_all_decodes:
        decodes = decodes[:eff_bs, :, :]
      else:
        decodes = decodes[:eff_bs, -1, :]
      out[f"{modality}-tokens"] = decodes
    elif modality == 'image' and decodes.shape[-1] == 1024:
      decodes = decodes - 2 # since 1 is the unk token.
      _num_decodes = decodes.shape[1]

      if return_all_decodes:
        decodes = decodes[:eff_bs, :, :].reshape(-1, 1024)
      else:
        decodes = decodes[:eff_bs, -1, :]

      img_decode = self.module.apply(
          {'params': params},
          method=self.module.decode_image_code,
          code_b=decodes)

      if return_all_decodes:
        decodes = decodes.reshape(eff_bs, _num_decodes, 1024)
        img_decode = img_decode.reshape((eff_bs, _num_decodes) + img_decode.shape[1:])

      out[f"{modality}-tokens"] = decodes
      out["image"] = img_decode

    elif modality == 'audio' and decodes.shape[-1] == 512:
      decodes = decodes - 2 # since 1 is the unk token.
      _num_decodes = decodes.shape[1]
      if return_all_decodes:
        decodes = decodes[:eff_bs, :, :].reshape(-1, 512)
      else:
        decodes = decodes[:eff_bs, -1, :]

      decodes = jnp.reshape(decodes, [-1, 32, 16])
      decodes = jnp.transpose(decodes, [0, 2, 1])
      decodes = jnp.reshape(decodes, [-1, 512])

      audio_decode = self.module.apply(
          {'params': params},
          method=self.module.decode_audio_code,
          code_b=decodes)

      if return_all_decodes:
        decodes = decodes.reshape(eff_bs, _num_decodes, 512)
        audio_decode = audio_decode.reshape((eff_bs, _num_decodes) + audio_decode.shape[1:])

      out[f"{modality}-tokens"] = decodes
      out["audio"] = audio_decode

    # We only use the aux output, return dummy values for the decodes output
    return jnp.ones((eff_bs, num_decodes, 1) if return_all_decodes else (eff_bs,1), dtype=jnp.int32), out

  @gin.configurable
  def predict_with_answer_options(self, params, batch, max_options=100, normalize_by_len=False):
    """Predict the most probable answer options"""
    choices = batch["choices"]  # [batch, n_choices, option_len]
    bs, n_options, choice_len = choices.shape
    encoded_inputs, encoder_masks = self.module.apply(
      {'params': params},
      batch,
      enable_dropout=False,
      method=self.module.encode
    )

    all_losses = []
    n_groups = (n_options + max_options - 1) // max_options
    for i in range(n_groups):
      group_choices = choices[:, i*max_options:(i+1)*max_options]
      group_num_options = group_choices.shape[1]
      group_choices = jnp.reshape(group_choices, [bs*group_num_options, -1])
      group_mask = group_choices > 0
      group_choices_batch = dict(
        targets=group_choices,
        inputs=jnp.pad(group_choices, [[0, 0], [1, 0]])[:, :-1],
        mask=group_mask,
      )
      group_choices_batch = dict(targets=dict(text=group_choices_batch))
      group_ex_encoded_inputs = [None if x is None else decoding.flat_batch_beam_expand(x, group_num_options)
                                 for x in encoded_inputs]
      group_ex_encoder_masks = decoding.flat_batch_beam_expand(encoder_masks, group_num_options)
      group_logits = self.module.apply(
        {'params': params},
        group_ex_encoded_inputs, group_ex_encoder_masks, group_choices_batch,
        method=self.module.decode,
      )
      group_logits, group_targets, group_mask = group_logits["text"]
      group_loss = compute_weighted_cross_entropy(
        group_logits, group_targets, group_mask, loss_normalizing_by_weight_sum=normalize_by_len, return_sum=False)[0]
      group_loss = group_loss * group_mask
      valid = jnp.any(jnp.reshape(group_mask, [bs, group_num_options, choice_len]) != 0, -1)
      group_loss = jnp.reshape(group_loss, [bs, group_num_options, choice_len])
      group_loss = jnp.sum(group_loss, -1)
      # Anything not valid get a very high loss so its not selected
      group_loss = group_loss + ((1 - valid) * 1e9)
      all_losses.append(group_loss)

    loss = jnp.concatenate(all_losses, -1)
    selected_option_ix = jnp.argmin(loss, -1)  # [bs, n_options]
    ix = jnp.arange(0, len(selected_option_ix))
    selected_loss = loss[ix, selected_option_ix]
    out = {
      "text-tokens": jnp.reshape(choices, [bs, n_options, -1])[ix, selected_option_ix],
      'scores': selected_loss,
      "all_scores": loss,
      "choice_ix": selected_option_ix
    }
    return jnp.ones((bs, 1), dtype=jnp.int32), out

  def score_batch(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
      return_sum: bool = True
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    raise NotImplementedError()


def compute_weighted_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    z_loss_param: float = 0.0,
    loss_normalizing_factor: Optional[float] = None,
    loss_normalizing_by_weight_sum: Optional[bool] = False, 
    return_sum: Optional[bool] = True, 
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute weighted cross entropy and entropy for log probs and targets.
  Args:
   logits: [batch, length, num_parallel ,num_classes] float array.
   targets: categorical targets [batch, length, num_parallel] int array.
   weights: None or array of shape [batch, length, num_parallel].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   z_loss_param: coefficient for auxiliary z-loss loss term.
   loss_normalizing_factor: Constant to divide loss by. If not specified, loss
     will not be normalized. Intended for backward compatibility with T5-MTF
     training. Should not normally be used.
  Returns:
    Tuple of scalar loss, z_loss, and weight sum.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  
  soft_targets = common_utils.onehot(targets, vocab_size, on_value=confidence, off_value=low_confidence)
  padded_targets = jnp.zeros(soft_targets.shape[:2] + (logits.shape[-1] - vocab_size,))
  soft_targets = jnp.concatenate([soft_targets, padded_targets], axis=-1)

  total_loss, total_z_loss = cross_entropy_with_logits(logits, soft_targets, z_loss=z_loss_param)
  total_loss = total_loss - normalizing_constant

  weight_sum = np.prod(targets.shape)
  if weights is not None:
    total_loss = total_loss * weights
    total_z_loss = total_z_loss * weights
    weight_sum = jnp.sum(weights)

  # By default, we do not normalize loss based on anything.
  # We don't normalize based on batch size because the optimizers we use are
  # pretty much scale invariant, so this simplifies things.
  # We don't normalize based on number of non-padding tokens in order to treat
  # each token as equally important regardless of sequence length.

  if loss_normalizing_by_weight_sum:
    total_loss /= jnp.maximum(weight_sum, 1)
    total_z_loss /= jnp.maximum(weight_sum, 1)
  else:
    if loss_normalizing_factor:
      total_loss /= loss_normalizing_factor
      total_z_loss /= loss_normalizing_factor
  
  if return_sum:
    return jnp.sum(total_loss), jnp.sum(total_z_loss), weight_sum
  else:
    return total_loss, total_z_loss, weight_sum


def multi_modality_loss(
    logits, z_loss_param=0.0,
    label_smoothing=None, loss_normalizing_factor=None,
    loss_normalizing_by_weight_sum=True,
):
  total_loss = 0
  metrics = {}

  for modality_name, (logit, target, mask) in logits.items():
    loss, z_loss, weight_sum = compute_weighted_cross_entropy(
        logit, target, mask, label_smoothing, z_loss_param,
        loss_normalizing_factor, loss_normalizing_by_weight_sum)
    total_loss += loss

    num_tokens = target.size
    if mask is None:
      nonpadding_tokens = np.prod(target.size)
    else:
      nonpadding_tokens = jnp.sum(mask)
    
    metrics.update({
      f'{modality_name}/accuracy': clu_metrics.Accuracy.from_model_output(
          logits=logit, labels=target.astype(jnp.int32), mask=mask),
      f'{modality_name}/loss': metrics_lib.AveragePerStep(total=loss),
      f'{modality_name}/nonpadding_fraction':
          clu_metrics.Average(total=nonpadding_tokens, count=num_tokens),
    })

    if z_loss is not None:
      metrics.update({
      f'{modality_name}/z_loss': metrics_lib.AveragePerStep(total=z_loss), 
      f'{modality_name}/cross_ent_loss': metrics_lib.AveragePerStep(total=loss - z_loss),
    })

  metrics.update({'loss': metrics_lib.AveragePerStep(total=total_loss)})
  return total_loss, metrics

