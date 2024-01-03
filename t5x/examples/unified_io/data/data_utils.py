import json
import logging
import functools
import logging
import re
from typing import Dict, List

import seqio
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.image_ops_impl import _ImageDimensions, _CheckAtLeast3DImage, _assert, \
  _is_tensor

import t5x.examples.unified_io.vocabularies as uio_vocab  # 0.0.15 version
from t5x.examples.unified_io import config
from t5x.examples.unified_io.config import VOCAB_START, NUM_DETECTION_BIN

# DEFAULT_EXTRA_IDS = 200 + 1000
DEFAULT_EXTRA_IDS = VOCAB_START + NUM_DETECTION_BIN  # 200 for denoising + 1000 extra ids

MODALITY_EXTRA_ID_N_FRAMES = 8  # 8 frames just in case
if MODALITY_EXTRA_ID_N_FRAMES:
    MODALITY_EXTRA_IDS = (1 + MODALITY_EXTRA_ID_N_FRAMES) * 2    # image/audio input + n * image/audio history
else:
    MODALITY_EXTRA_IDS = 0

uio_vocab.PAD_ID = 0


def get_default_vocabulary():
  if config.TOKENIZER == "t5x-v1":
    return seqio.SentencePieceVocabulary(
      "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model", 1000)
  elif config.TOKENIZER == "llama":
    if not config.LLAMA_TOKENIZER_PATH:
      raise ValueError("`config.LLAMA_TOKENIZER_PATH` should point to the LLAMA tokenizer`")
    return uio_vocab.SentencePieceVocabulary(
      config.LLAMA_TOKENIZER_PATH,
      extra_ids=DEFAULT_EXTRA_IDS,
      reverse_extra_ids=True,
      modality_extra_id_n_frames=MODALITY_EXTRA_ID_N_FRAMES,
      hack_to_t5_start_tokens=True,
      prefix_as_special_token=True,
    )
  else:
    raise ValueError(config.TOKENIZER)


@seqio.map_over_dataset
def tokenize(data: Dict, copy_pretokenized=False):
  voc = get_default_vocabulary()
  for k in ["text_inputs", "text_targets"]:
    if k in data:
      v = data[k]
      if copy_pretokenized:
        data[f'{k}_pretokenized'] = v
      data[k] = voc.encode_tf(v)
  return data


def validate_keyword_prompts(prompts):
  """Check prompts have a consistent set of keywords"""
  all_keywords = [sorted(re.findall("{([^{}]*)}", x, re.M)) for x in prompts]
  keywords = all_keywords[0]
  assert len(keywords) == len(set(keywords))
  assert all(keywords == x for x in all_keywords), f"Inconsistent keywords in prompts {all_keywords}"
  assert not any("{" not in word[1:-1] and "}" in word[1:-1] for word in keywords)


def apply_keyword_prompt(prompt, allow_missing=False, **kwargs):
  """Fills in the brackted keywords in `prompt` with the keywords in `kwargs`"""
  return valid_regex_replace(prompt, {"{"+k+"}": v for k, v in kwargs.items()}, allow_missing)


def valid_regex_replace(prompt, replacements, allow_missing=False):
  """Replaces occurrences of keys in `replacements` in `prompt` with the values
  Assume occurrences only occur once."""
  if allow_missing:
    for c, value in replacements.items():
      res = tf.strings.split(prompt, c)
      if tf.shape(res) == 2:
        prompt = tf.strings.join([res[0], value, res[1]])
    return prompt
  else:
    for c, value in replacements.items():
      # We avoid regex_replace since it has issues if the replacement has
      # bashslashes that appears can't be avoided
      res = tf.strings.split(prompt, c)
      tf.assert_equal(tf.shape(res), 2, message="prompt substitution error")
      prompt = tf.strings.join([res[0], value, res[1]])
    return prompt


def trim_or_pad_tf(x, seq_len, pad_constant=0):
  """Trim or pad tensorflow tensor `x` to `seq_len`"""
  x = x[:seq_len]
  sh = list(x.shape)
  sh[0] = seq_len
  x = tf.pad(
    x,
    [[0, seq_len-tf.shape(x)[0]]] + [[0, 0]]*(len(sh)-1),
    constant_values=pad_constant,
  )
  return tf.ensure_shape(x, sh)


def trim_or_pad_tf_2d(x, batch, seq_len):
  """Trim or pad tensorflow tensor `x` to `batch` amd `seq_len`"""
  x = x[:batch, :seq_len]
  sh = [batch, seq_len] + list(x.shape)[2:]
  x = tf.pad(x,
             [[0, batch-tf.shape(x)[0]]] +
             [[0, seq_len-tf.shape(x)[1]]] +
             [[0, 0]]*(len(sh)-2))
  return tf.ensure_shape(x, sh)


def normalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  """Normalizes the image to zero mean and unit variance."""
  offset = tf.constant(offset)
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= tf.cast(offset, image.dtype)

  scale = tf.constant(scale)
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= tf.cast(scale, image.dtype)
  return image


def unnormalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  scale = tf.cast(tf.expand_dims(tf.expand_dims(tf.constant(scale), axis=0), axis=0), image.dtype)
  image *= scale

  offset = tf.cast(tf.expand_dims(tf.expand_dims(tf.constant(offset), axis=0), axis=0), image.dtype)
  image += offset
  return image


def denormalize_boxes(boxes, image_shape):
    """Converts boxes normalized by [height, width] to pixel coordinates.
    Args:
      boxes: a tensor whose last dimension is 4 representing the coordinates of
        boxes in ymin, xmin, ymax, xmax order.
      image_shape: a list of two integers, a two-element vector or a tensor such
        that all but the last dimensions are `broadcastable` to `boxes`. The last
        dimension is 2, which represents [height, width].
    Returns:
      denormalized_boxes: a tensor whose shape is the same as `boxes` representing
        the denormalized boxes.
    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
    with tf.name_scope('denormalize_boxes'):
      if isinstance(image_shape, list) or isinstance(image_shape, tuple):
        height, width = image_shape
        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)
      else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        height, width = tf.split(image_shape, 2, axis=-1)

      ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
      ymin = ymin * height
      xmin = xmin * width
      ymax = ymax * height
      xmax = xmax * width

      denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
      return denormalized_boxes


def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.
  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.
  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('clip_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      max_length = [height, width, height, width]
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.unstack(image_shape, axis=-1)
      max_length = tf.stack(
          [height, width, height, width], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes


def get_non_empty_box_indices(boxes):
    """Get indices for non-empty boxes."""
    # Selects indices if box height or width is 0.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(
        tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
    return indices[:, 0]


def resize_and_crop_boxes(boxes, image_scale, output_size, offset, paddings):
    """Resizes boxes to output size with scale and offset."""
    # Adjusts box coordinates based on image_scale, offset and paddings.
    boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    boxes += tf.tile(tf.expand_dims(paddings, axis=0), [1, 2])
    # Clips the boxes.
    boxes = clip_boxes(boxes, output_size)
    return boxes


def resize_and_pad_default(
    image, is_training, is_input=True, masks=None, boxes=None, box_labels=None,
    random_scale_min=None, random_scale_max=None, random_scale_ratio=None,
    resize_method=None, is_history=False
):
  """Apply `resize_and_pad` with default settings"""
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if random_scale_min is None:
    random_scale_min = config.RANDOM_SCALE_MIN
  if random_scale_max is None:
    random_scale_max = config.RANDOM_SCALE_MAX
  if random_scale_ratio is None:
    random_scale_ratio = config.RANDOM_SCALE_RATIO
  if resize_method is None:
    resize_method ='random' if is_training else tf.image.ResizeMethod.BILINEAR
  if len(image.shape) == 4 or is_history:
    assert is_input
    output_size = config.IMAGE_HISTORY_INPUT_SIZE
  else:
    output_size = config.IMAGE_INPUT_SIZE if is_input else config.IMAGE_TARGET_SIZE
  return resize_and_pad(
    image, output_size,
    masks, boxes, box_labels,
    random_scale_min=random_scale_min,
    random_scale_max=random_scale_max,
    do_random_scale=is_training,
    random_scale_ratio=random_scale_ratio,
    resize_method=resize_method
  )


def resize_and_pad(image, desired_output_size, masks=None, boxes=None, box_labels=None,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   shrink_both_sides=True, filter_box=True,
                   desired_target_size=None, random_scale_ratio=0.0,
                   resize_method=tf.image.ResizeMethod.BILINEAR,
                   pad_value=0):
    """Resizes and pads an input image/video to `desired_output_size`

    Support random scaling augmentation if `do_random_scale` is True

    If `masks` or `boxes` are given, the same transformation that is applied ot the image
    is applied to them. Boxes can be completely removed if doing scaling augmentation, in which
    case the deleted boxes will not be returned.

    outputs:
      image: The resized image/video
      image_mask: A mask showing which pixels are padding in the output image
      meta-data: Meta-data about the transformation and the boxes/masks that were also transformed
    """
    desired_height, desired_width = desired_output_size
    desired_height_f = tf.cast(desired_height, dtype=tf.float32)
    desired_width_f = tf.cast(desired_width, dtype=tf.float32)

    is_video = len(image.shape) == 4

    if is_video:
      height = tf.cast(tf.shape(image)[1], tf.float32)
      width = tf.cast(tf.shape(image)[2], tf.float32)
    else:
      height = tf.cast(tf.shape(image)[0], tf.float32)
      width = tf.cast(tf.shape(image)[1], tf.float32)

    if boxes is not None:
        # Converts boxes from normalized coordinates to pixel coordinates.
        # Now the coordinates of boxes are w.r.t. the original image.
        boxes = denormalize_boxes(boxes, [height, width])

    if do_random_scale:
        random_scale_factor = tf.random.uniform([], random_scale_min, random_scale_max)
        if not shrink_both_sides:
            # Max random is where scale * W > W_desired
            #                     scale * H > H_desired
            rsf_max = tf.maximum(desired_width_f / width, desired_height_f / height)
            random_scale_factor = tf.minimum(rsf_max, random_scale_factor)

        scaled_y = tf.cast(random_scale_factor * desired_height_f, tf.int32)
        scaled_x = tf.cast(random_scale_factor * desired_width_f, tf.int32)

        # Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_y = tf.cast(scaled_y, tf.float32) / height
        image_scale_x = tf.cast(scaled_x, tf.float32) / width
        
        image_scale = tf.cond(tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(random_scale_ratio, tf.float32)), 
            lambda: tf.maximum(image_scale_x, image_scale_y),
            lambda: tf.minimum(image_scale_x, image_scale_y))
        
        # Don't scale any side lower than to 64
        # For very wide images, this truncates the edge in order to keep the resolution
        # reasonable
        image_scale = tf.maximum(image_scale, 64.0 / tf.minimum(height, width))

        # Select non-zero random offset (x, y) if scaled image is larger than
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.cast(scaled_height - desired_height, tf.float32)
        offset_x = tf.cast(scaled_width - desired_width, tf.float32)
        offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
        offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
        offset_y = tf.cast(offset_y, tf.int32)
        offset_x = tf.cast(offset_x, tf.int32)
    else:
        image_scale_y = desired_height_f / height
        image_scale_x = desired_width_f / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.constant(0)
        offset_x = tf.constant(0)

    # Now resize and crop
    if resize_method == 'random' and do_random_scale and (not tf.executing_eagerly()):
        resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
        # print("Random resize method:\n{}".format(','.join(resize_methods)))
        image = apply_with_random_selector(
            image,
            lambda x, method_idx: tf.image.resize(x, [scaled_height, scaled_width],
                                                  tf.image.ResizeMethod.__dict__[resize_methods[method_idx]],
                                                  antialias=True),
            num_cases=len(resize_methods))

    elif resize_method != 'random':
        image = tf.image.resize(image, [scaled_height, scaled_width], method=resize_method, antialias=True)
    else:
        logging.info(f"you passed in {resize_method} but doing bilinear resize instead (possibly because eager is on or evaluation is on.)")
        image = tf.image.resize(image, [scaled_height, scaled_width],
                                method=tf.image.ResizeMethod.BILINEAR, antialias=True)

    image = tf.clip_by_value(image, 0.0, 1.0)
    
    if is_video:
        # frames x H x W x C
        image = image[:,offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
        H = tf.shape(image)[1]
        W = tf.shape(image)[2]
    else:
        # H x W x C
        image = image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
        H = tf.shape(image)[0]
        W = tf.shape(image)[1]

    if config.PAD_ONE_SIDE:
      top_pad = 0
      left_pad = 0
    else:
      top_pad = (desired_height - H) // 2
      left_pad = (desired_width - W) // 2

    # Get the mask which indicates which regions were padded
    mask = tf.ones(tf.concat([tf.shape(image)[:-1], [1]], 0), dtype=tf.int32)
    image_mask = tf.squeeze(pad_to_bounding_box(
      mask, top_pad, left_pad, desired_height, desired_width), -1)

    image = pad_to_bounding_box(image, top_pad, left_pad, desired_height, desired_width,
                                value=pad_value)

    if is_video:
        image.set_shape([None, desired_height, desired_width, 3])
    else:
        image.set_shape([desired_height, desired_width, 3])

    if masks is not None and tf.size(masks) != 0:
      masks = tf.image.resize(
        masks, [scaled_height, scaled_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      if len(masks.shape) == 3:
        masks = masks[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]                    
      else:
        masks = masks[:, offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]                    

      masks = pad_to_bounding_box(masks, top_pad, left_pad, desired_height, desired_width)
      masks = tf.image.resize(masks, desired_target_size,
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
    indices = None
    if boxes is not None:
      boxes = resize_and_crop_boxes(
          boxes, 
          tf.stack([image_scale, image_scale]), 
          [desired_height, desired_width], 
          tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32),
          tf.cast(tf.stack([top_pad, left_pad]), dtype=tf.float32))

      if filter_box:
        indices = get_non_empty_box_indices(boxes)
      else:
        indices = tf.range(tf.shape(boxes)[0])
      boxes = tf.gather(boxes, indices)
      
      if box_labels is not None:
        box_labels = tf.gather(box_labels, indices)

    # Stores meta meta-data about how the image was resized, needed if we want
    # reverse the padding/resizing later
    image_info = tf.stack([
        tf.cast(top_pad, tf.float32),
        tf.cast(left_pad, tf.float32),
        1.0 / image_scale,
        height,
        width,
        tf.cast(offset_y, dtype=tf.float32) / height,
        tf.cast(offset_x, dtype=tf.float32) / width,
        tf.cast(offset_y, dtype=tf.float32),
        tf.cast(offset_x, dtype=tf.float32),
        tf.cast(scaled_height, dtype=tf.float32),
        tf.cast(scaled_width, dtype=tf.float32),
    ])

    outputs = (image_info, masks, boxes, box_labels, indices)
    return image, image_mask, outputs


def tokens_to_values(tokens):
  """Convert location tokens to floating point values within [0, 1]"""
  eos = tf.where(tokens == 1)
  if tf.shape(eos)[0] > 0:
    tokens = tokens[:eos[0, 0]]
  vals = tokens - 32000
  vals = vals[vals >= 0]
  vals = vals[vals < config.NUM_DETECTION_BIN]
  return 1.0 - vals / (config.NUM_DETECTION_BIN - 1)


def values_to_tokens(vals, clss=None):
  """Convert floating point values in [0, 1] to tokens"""
  vals = tf.convert_to_tensor(vals)
  num_bins = config.NUM_DETECTION_BIN
  vocab_start = config.VOCAB_START
  quantized_boxes = tf.cast(vals * (num_bins-1), tf.int32)

  # For values that were exactly one
  vals = tf.constant([f'<extra_id_{i}>' for i in range(vocab_start, vocab_start+num_bins)])
  tokens = tf.gather(vals, quantized_boxes)

  if clss is not None:
    tokens = tf.concat([tokens, tf.expand_dims(clss, 1)], axis=-1)

  return tokens


def convert_bboxes_to_str(
  boxes,
  labels=None,
  image_size=config.IMAGE_INPUT_SIZE[0],
  convert_to_str=True,
  shuffle=True,
  seperator=" "
):
  """Converts a sequence of bound boxes into a sequence of location tokens"""
  if shuffle:
    # shuffle the labels, ids.
    indices = tf.range(start=0, limit=tf.shape(boxes)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    if labels is not None:
      labels = tf.gather(labels, shuffled_indices)
    boxes = tf.gather(boxes, shuffled_indices)

  boxes_str = values_to_tokens(tf.cast(boxes, tf.float32)/image_size)

  if labels is not None:
    labels_str = tf.expand_dims(labels, axis=-1)
    boxes_str = tf.concat([boxes_str, labels_str], axis=-1)

  if convert_to_str:
    boxes_str = tf.strings.reduce_join(boxes_str, separator=' ', axis=-1)
    boxes_str = tf.strings.reduce_join(boxes_str, separator=seperator)
  return boxes_str


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
    func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
    for case in range(num_cases)])[0]


def _stateless_shuffle(x: tf.Tensor, seed):
  if hasattr(tf.random.experimental, 'stateless_shuffle'):
    return tf.random.experimental.stateless_shuffle(x, seed=seed)
  else:
    vals = tf.random.stateless_uniform(tf.shape(x), seed)
    ixs = tf.argsort(vals)
    return tf.gather(x, ixs)


def sample_patches(mask, n_patches, stateless=False, seeds=None):
  input_sample_valid = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
  input_sample_masked = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask == 0)
  if stateless:
    encoder_pos_ids = tf.concat([
      _stateless_shuffle(input_sample_valid, seeds[0]),
      _stateless_shuffle(input_sample_masked, seeds[1])], axis=0)[:n_patches]
  else:
    encoder_pos_ids = tf.concat([
      tf.random.shuffle(input_sample_valid),
      tf.random.shuffle(input_sample_masked)], axis=0)[:n_patches]
  encoder_pos_ids = tf.reshape(encoder_pos_ids, (n_patches,))
  encoder_pos_ids = tf.cast(encoder_pos_ids, tf.int32)
  return encoder_pos_ids


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, value=0):

  return pad_to_bounding_box_internal(
      image,
      offset_height,
      offset_width,
      target_height,
      target_width,
      check_dims=True, 
      value=value)


def pad_to_bounding_box_internal(image, offset_height, offset_width,
                                 target_height, target_width, check_dims, value):

  with ops.name_scope(None, 'pad_to_bounding_box_with_one_internal', [image]):
    image = ops.convert_to_tensor(image, name='image')

    is_batch = True
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError(
          '\'image\' (shape %s) must have either 3 or 4 dimensions.' %
          image_shape)

    batch, height, width, depth = _ImageDimensions(image, rank=4)

    after_padding_width = target_width - offset_width - width

    after_padding_height = target_height - offset_height - height

    if check_dims:
      assert_ops = _CheckAtLeast3DImage(image, require_static=False)
      assert_ops += _assert(offset_height >= 0, ValueError,
                            'offset_height must be >= 0')
      assert_ops += _assert(offset_width >= 0, ValueError,
                            'offset_width must be >= 0')
      assert_ops += _assert(after_padding_width >= 0, ValueError,
                            'width must be <= target - offset')
      assert_ops += _assert(after_padding_height >= 0, ValueError,
                            'height must be <= target - offset')
      image = control_flow_ops.with_dependencies(assert_ops, image)

    # Do not pad on the depth dimensions.
    paddings = array_ops.reshape(
        array_ops.stack([
            0, 0, offset_height, after_padding_height, offset_width,
            after_padding_width, 0, 0
        ]), [4, 2])
    padded = array_ops.pad(image, paddings, constant_values=value)

    padded_shape = [
        None if _is_tensor(i) else i
        for i in [batch, target_height, target_width, depth]
    ]
    padded.set_shape(padded_shape)

    if not is_batch:
      padded = array_ops.squeeze(padded, axis=[0])

    return padded


def random_element(vec, seed=None):
  if isinstance(vec, list):
    if len(vec) == 1:
      return vec[0]
    assert len(vec) > 0
    vec = tf.constant(vec)
  if seed is not None:
    ix = tf.random.stateless_uniform((), seed, 0, tf.shape(vec)[0], tf.int32)
  else:
    ix = tf.random.uniform((), 0, tf.shape(vec)[0], tf.int32)
  return vec[ix]


def load_class_names(path, cache={}) -> List[str]:
  if path not in cache:
    cache[path] = _load_class_name(path)
  return cache[path]


def _load_class_name(path) -> List[str]:
  with open(path) as f:
    data = json.load(f)
  if isinstance(data, dict):
    # Assume we have a name: int mapping instead of just a list
    data = {int(k): v for k, v in data.items()}
    keys = sorted(data)
    if path in {"metadata/imagenet/imagenet_2012_class_name.json", "metadata/ava/ava_action_v2.2.json"}:
      # Meta-data is expected to be offset by one
      assert keys == list(range(1, len(data)+1))
    else:
      # Should be 0 indexed
      assert keys == list(range(len(data)))
    data = [data[k] for k in keys]
  else:
    assert isinstance(data, list)
  return data


def box_mask(box, image_size, inclusive=False):
  """
  build a `image_size` mask with ones for pixels within `box`

  box: [4,] integer tensors for y1, x1, y2, x2
  box_size: [2] integer sizes
  inclusive whether the box includes the x2/y2 pixel in `box`
  """
  y, x = image_size
  ymin, xmin, ymax, xmax = tf.unstack(box)
  w = xmax - xmin
  h = ymax - ymin
  if inclusive:  # +1 to include the `ymax` and `xmax` pixel
    w += 1
    h += 1
  mask = tf.pad(
    tf.ones([h, w], dtype=tf.int32), [
      [ymin, y-(ymin + h)],
      [xmin, x-(xmin + w)],
    ]
  )
  return tf.ensure_shape(mask, [y, x])
