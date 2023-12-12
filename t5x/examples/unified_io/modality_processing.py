"""Code for handling modalities"""

from collections import OrderedDict
from typing import Mapping

from flax import traverse_util
from seqio import TaskRegistry, FeatureConverter

from t5x.examples.unified_io.audio_encoder import AudioEncoder
from t5x.examples.unified_io.image_encoder import ImageEncoder
from t5x.examples.unified_io.input_modalities import *
from t5x.examples.unified_io.target_modalities import *


@gin.configurable
def get_target_modalities(
    target_modality=['text', 'image', 'audio'],
    image_vae_config: ImageViTVQGANConfig=VAEConfig(),
    audio_vae_config: AudioViTVQGANConfig=AudioViTVQGANConfig(),
  ) -> Dict[str, ModalityEncoder]:
  """Return the encoders to use for target modalities"""

  out = {}
  if 'text' in target_modality:
    out['text'] = TargetTextEncoder()
  if 'image' in target_modality:
    out['image'] = TargetImageDVAEEmbedder(image_vae_config)
  if 'audio' in target_modality:
    out['audio'] = TargetAudioDVAEEmbedder(audio_vae_config)
  return out


@gin.configurable
def get_input_modalities(
  input_modality=('text', 'image', 'image_history', 'audio', 'audio_history'),
  image_vit_cfg: ImageVitFeatureConfig=ImageVitFeatureConfig(),
  audio_vit_cfg: AudioVitFeatureConfig=AudioVitFeatureConfig(),
  image_history_cfg: ImageResamplerConfig=ImageResamplerConfig(),
  audio_history_cfg: AudioResamplerConfig=AudioResamplerConfig(),
  max_img_history=None,
  max_audio_history=None,
  use_image_vit = False,
  use_audio_vit = False,
  freeze_vit=False,
  use_image_history_vit = False,
  use_audio_history_vit = False,
) -> Dict[str, ModalityEncoder]:
  """Returns the ModalityEncoder for the input modalities"""

  out = dict()
  if 'text' in input_modality: 
    out["text"] = InputTextEncoder()

  image_encoder = None
  if use_image_vit or use_image_history_vit:
    image_encoder = ImageEncoder(image_vit_cfg)

  audio_encoder = None
  if use_audio_vit or use_audio_history_vit:
    audio_encoder = AudioEncoder(audio_vit_cfg)

  if 'image' in input_modality:
    out["image"] = InputImageViTEncoder(
      image_encoder if use_image_vit else None, freeze_vit)
  
  if 'image_history' in input_modality:
    out["image_history"] = InputImageHistoryViTEncoder(
      image_encoder if use_image_history_vit else None, image_history_cfg, max_img_history)

  if 'audio' in input_modality:
    out["audio"] = InputAudioViTEncoder(audio_encoder if use_audio_vit else None)
  
  if 'audio_history' in input_modality:
    out["audio_history"] = InputAudioHistoryViTEncoder(
      audio_encoder if use_audio_history_vit else None, audio_history_cfg, max_audio_history)
  return out


class UnifiedIOFeatureConverter(FeatureConverter):
  """seqio.FeatureConverter for UIO 2

  This just applies the `ModalityEncoder` returned by the above functions
  """
  TASK_FEATURES = {}
  MODEL_FEATURES = {}

  def __init__(self, pack=False, use_custom_packing_ops=False):
    assert not pack
    assert not use_custom_packing_ops
    super().__init__(pack=False, use_custom_packing_ops=False)

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    raise NotImplementedError()

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    raise NotImplementedError()

  def __call__(self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      features = traverse_util.unflatten_dict(features, sep="/")

      # Update input features using the modality-specific converter methods
      converted_input_features = {}
      for k, v in get_input_modalities().items():
        converted_input_features[k] = v.convert_inputs(
          features["inputs"].get(k), task_feature_lengths)

      converted_target_features = {}
      for k, v in get_target_modalities().items():
        converted_target_features[k] = v.convert_inputs(
          features["targets"].get(k), task_feature_lengths)

      output_features = dict(
        inputs=converted_input_features,
        targets=converted_target_features
      )

      # Special cases that might need to be used inference
      if "choices" in features:
        output_features["choices"] = get_target_modalities()["text"].convert_choices(
          features["choices"], task_feature_lengths)
      for k in ["image_info", "box"]:
        if k in features:
          output_features[k] = features[k]
      return traverse_util.flatten_dict(output_features, keep_empty_nodes=True, sep="/")

    return ds.map(convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@seqio.map_over_dataset
def unified_io_preprocessor(features, output_features, sequence_length):
  """General pre-processing function that builds models features from multi-modal inputs.

  This function should be used as the last pre-processor for all tasks, it calls the
  modality-specific preprocess modules produced by `get_input_modalities` and
  `get_target_modalities`to apply model-specific preprocess that needs to be done before
  the tasks are combined into a mixture.

  Args:
    features: dictionary with a subset of the following fields:
      text_inputs: int32 array of tokenized text inputs (without EOS) or tf.string scalar,
                   the prompt/input text to the model
      text_targets: int32 array tokenized of text inputs (without EOS) or
                    tf.string scalar, the the output text to generate.
                    Can also be a list of string tensors or ragged int32 tensor to represent
                    multiple correct answer options
      image_inputs: RGB image size `IMAGE_INPUT_SIZE`  in float32 format, the input image
      image_input_masks: image_mask for size `IMAGE_INPUT_SIZE` marking pixels to
                         included iff `image_inputs` is included
      audio_inputs: Audio spectrogram [256, 128]
      audio_inputs_masks: Audio spectrogram mask
      video_inputs: RGB by time video in float32 format
      video_inputs_masks: 2D mask of the same height/width of the video
      audio_history_inputs: Audio spectrogram history [N, 256, 128]
      audio_history_input_masks: Masks for audio_history_inputs
      image_targets: (optional) RGB image of `IMAGE_TARGET_SIZE` in float32 format, the target
                     image to generate
      image_target_masks: (optional) image_mask for size `IMAGE_TARGET_SIZE` or `IMAGE_INPUT_SIZE`
                         included iff `image_targets` is included.
                         If of `IMAGE_INPUT_SIZE`, the mask will be applied as if re-scaled to
                         `IMAGE_TARGET_SIZE`, but we can avoid some compute/rounding errors by
                         avoiding explicitly rescaling it in this case.
      audio_targets: Audio spectrogram target
      audio_targets_masks: Target audio mask

      eval: sub-dictionary of features that should be passed to metric functions
            (e.g., ground truth labels)
      choices: List of strings or ragged int32 tensor of text answer options
  """
  input_features = {}
  input_modalities = get_input_modalities()
  for k, v in input_modalities.items():
    input_features[k] = v.preprocess_inputs(features, output_features, sequence_length)

  target_features = {}
  for k, v in get_target_modalities().items():
    target_features[k] = v.preprocess_inputs(features, output_features, sequence_length)

  # Features that might be needed by metric functions or for evaluations
  if "meta" in features:
    meta = features["meta"]
  else:
    meta = {}
  for k in features:
    if k.startswith("meta/"):
      meta[k] = features[k]

  out = dict(
    inputs=input_features,
    targets=target_features,
    meta=meta
  )

  # If there are answer choices, they need to be passed through to the model
  if "choices" in features:
    out["choices"] = features["choices"]

  out = traverse_util.flatten_dict(out, keep_empty_nodes=False, sep="/")
  return out


OUTPUT_FEATURES = {}


def _build_output_features():
  for kind, src in [
    ("inputs", get_input_modalities()),
    ("targets", get_target_modalities())
  ]:
    for name, encoder in src.items():
      for feature_name, fe in encoder.get_output_features().items():
        OUTPUT_FEATURES[f"{kind}/{name}/{feature_name}"] = fe


_build_output_features()


def _update_outputs(_config):
  # Hack to update Task output_features to match the modality encoders registered via. gin
  # We need this since we register the Tasks at import time, before gin bindings are processed,
  # but the output features of the tasks might change depending on the gin configuration
  OUTPUT_FEATURES.clear()
  _build_output_features()
  items = sorted(list(OUTPUT_FEATURES.items()))
  for name, task in TaskRegistry._REGISTRY.items():
    task._output_features = OrderedDict(items)
  return {}


gin.config.register_finalize_hook(_update_outputs)


def get_input_spec(batch_size=1, seq_lens=None):
  """Gets the shape/types of the input batch the model expects"""
  if seq_lens is None:
    # Dummy lengths that can still be used to initial the model
    seq_lens = {
      "is_training": False, "text_inputs": 1, "text_targets": 1, "image_input_samples": 1,
      "image_history_input_samples": 1, "audio_input_samples": 1,
      "audio_history_input_samples": 1, 'num_frames': 1,
    }
  data = tf.data.Dataset.from_tensors(dict(text_inputs="a", text_targets="b"))
  data = unified_io_preprocessor(data, OUTPUT_FEATURES, seq_lens)
  data = UnifiedIOFeatureConverter()(data, seq_lens)
  input_types = {k: v.dtype.as_numpy_dtype() for k, v in data.element_spec.items()}
  input_shapes = {k: ([batch_size] + list(v.shape)) for k, v in data.element_spec.items()}
  return input_shapes, input_types