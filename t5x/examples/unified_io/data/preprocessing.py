"""Preprocessing functions used in seqio.Tasks"""
from functools import reduce
from typing import Dict, List, Any
import gin
import tensorflow as tf
import seqio
import numpy as np

from t5x.examples.unified_io.data import data_utils
from t5x.examples.unified_io.data.prompt_definition import Prompt

from t5x.examples.unified_io.data.data_utils import resize_and_pad, resize_and_pad_default, \
  random_element, convert_bboxes_to_str, valid_regex_replace, load_class_names

from t5x.examples.unified_io import config
from t5x.examples.unified_io.data.prompt_dict import Image_Generation_No_Text


@seqio.utils.map_over_dataset
def rekey(x: Dict[str, Any], key_map: Dict[str, List]):
  """Get elements from possibly nested dict `x` according to the mapping in `key_map`.

  Args:
      x: an example to process
      key_map: dictionary mapping new keys to original keys
  Returns:
      A preprocessed example with the format listed above.
  """
  def _get(data, keys):
    return reduce(dict.get, keys, data)

  return {
    new_key: _get(x, old_key) if old_key else ''
    for new_key, old_key in key_map.items()
  }


def flatten_parts(ds: tf.data.Dataset, parts: List[str], add_index=False) -> tf.data.Dataset:
  """Flatten `ds` so that the features in `parts` are flattened (meaning each slice of those
  features becomes an individual example) and the other features in ds are duplicated"""
  def _flatten(ex):
    flat_key = {k: ex[k] for k in parts}
    if add_index:
      flat_key['index'] = tf.range(len(ex[parts[0]]))

    flat_ds = tf.data.Dataset.from_tensor_slices(flat_key)

    def _merge(_flat_ex):
      for k, v in ex.items():
        if k not in parts:
          _flat_ex[k] = v
      return _flat_ex
    return flat_ds.map(_merge)

  return ds.flat_map(_flatten)


def flatten_by_label(ds: tf.data.Dataset, target_keys) -> tf.data.Dataset:
  """Flatten `ds` so that each unique label becomes an example"""

  def _flatten(ex):
    labels = ex["label"]
    unique_label, idx = tf.unique(labels)
    def _get_labels(label):
      select = labels == label
      out = dict(ex)
      for k in target_keys:
        if k == 'bbox':
          out[k] = tf.reshape(out[k], [-1, 4])
        out[k] = out[k][select]
      return out
    return tf.data.Dataset.from_tensor_slices(unique_label).map(_get_labels)
  return ds.flat_map(_flatten)


@gin.configurable
def refer_expression_preprocessor(
    ds, sequence_length, dataset_name, box_format="yxyx", label_boxes=False):
  is_training = sequence_length.get('is_training', True)

  @seqio.map_over_dataset(num_seeds=2)
  def build_features(ex, seeds):
    img = ex["image"]
    box = ex["bbox"]
    if box_format == "xyxy":
      box = tf.stack([box[1], box[0], box[3], box[2]])
    boxes = tf.expand_dims(box, 0)
    labels = ex["label"]
    img, img_mask, this_image_info = resize_and_pad_default(img, is_training, boxes=boxes)

    image_info, masks, boxes, _, indices = this_image_info

    if tf.shape(boxes)[0] > 0:
      if len(labels.shape) > 0:
        # Box has multiple labels, just select one at random
        exp_ix = tf.random.stateless_uniform((), seeds[0], 0, tf.shape(labels)[0], tf.int32)
        labels = labels[exp_ix]
      labels = tf.strings.lower(labels)

      # convert the boxes and labels text inputs/targets
      text_targets = convert_bboxes_to_str(boxes, labels if label_boxes else None)

      prompt_list = Prompt().get_prompt_list('Refexp', dataset_name)
      prompt = random_element(prompt_list, seeds[1])
      text_inputs = valid_regex_replace(prompt, {"{}": labels})
      text_inputs = tf.strings.join(["[Text] [S] ", text_inputs])
      if len(ex["refexp_id"].shape) > 0:
        example_id = ex["refexp_id"][exp_ix]
      else:
        example_id = ex["refexp_id"]
      valid = True
    else:
      # In rare cases the bounding box can be removed during pre-processing,
      # we just skip the example in that case
      example_id = tf.convert_to_tensor(-1, tf.int64)
      text_inputs = tf.constant("")
      text_targets = tf.constant("")
      valid = False

    return dict(
      image_inputs=img,
      image_input_masks=img_mask,
      text_inputs=text_inputs,
      text_targets=text_targets,
      valid=valid,
      meta=dict(
        example_id=example_id,
        image_info=image_info,
        boxes=boxes,
        src_boxes=ex["bbox"],
      )
    )

  ds = flatten_parts(ds, ["bbox", "label", "refexp_id"])
  ds = build_features(ds)
  ds = ds.filter(lambda x: x["valid"])
  return ds


@gin.configurable()
def image_generation_preprocessor(ds, sequence_length, dataset_name, random_gen_ratio=0.1):
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    img = ex['image']
    if img.dtype == tf.string:
      img = tf.image.decode_jpeg(img, channels=3)

    image_targets, img_mask, meta = resize_and_pad_default(
      img, is_training, is_input=False)
    image_info = meta[0]

    if tf.random.uniform([], 0, 1, dtype=tf.float32) < random_gen_ratio:
      # Classifier free guidance prompt
      text_inputs = random_element(Image_Generation_No_Text)
      sampled_caption = ''
    else:
      if len(ex["captions"].shape) == 1:
        # Multiple captions associated with the image, pick one at random
        sampled_caption = random_element(ex["captions"])
      else:
        sampled_caption = ex['captions']

      sampled_caption = tf.strings.lower(sampled_caption)

      sampled_caption = tf.strings.regex_replace(sampled_caption, r"\\", " ")
      sampled_caption = tf.strings.lower(sampled_caption)

      prompt = random_element(Prompt().get_prompt_list('Image_Generation', dataset_name))
      text_inputs = valid_regex_replace(prompt, {"{}": sampled_caption})

    text_inputs = tf.strings.join(["[Image] [S] ", text_inputs])

    meta = {
      'image_info': image_info,
      'original_query': sampled_caption,
    }
    if "image/filename" in ex:
      meta["image/filename"] = ex["image/filename"]

    return {
      "meta": meta,
      'image_targets': image_targets,
      'image_target_masks': img_mask,
      'text_inputs': text_inputs,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def image_caption_preprocessor(
    ds, sequence_length, dataset_name,
    multiple_targets=False, localized_narrative=False):
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex):
    img = ex['image']
    if img.dtype == tf.string:
      img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, img_meta = resize_and_pad_default(img, is_training)

    if localized_narrative:
      prompt_list = Prompt().get_prompt_list('Image_Localized_Narrative', dataset_name)
    else:
      prompt_list = Prompt().get_prompt_list('Image_Captioning', dataset_name)

    text_inputs = tf.strings.join(["[Text] [S] ", random_element(prompt_list)])

    captions = ex["captions"]
    text_targets = tf.strings.lower(captions)
    if len(captions.shape) != 0:
      if multiple_targets:
        # `text_targets` is a list of strings, model will be trained to predict all at once
        text_targets = tf.strings.lower(text_targets)
        text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")
      else:
        # Just train on a random string
        text_targets = random_element(captions)

    # We have added these to help normalize CC3M or other noisy captions where sometimes the
    # entire caption is in quotes. Probably should not be applied to every captioning dataset,
    # but that is how UIO 2 was trained we leave it
    text_targets = tf.strings.regex_replace(text_targets, r"\\", " ")
    text_targets = tf.strings.regex_replace(text_targets, r"<person>", "person")

    meta = {
      'all_references': ex["captions"],
    }
    if "image/filename" in ex:
      meta["example_id"] = ex["image/filename"]
    return {
      'meta': meta,
      'image_inputs': img,
      'image_input_masks': img_mask,
      'text_inputs': text_inputs,
      'text_targets': text_targets,
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def image_inpainting_preprocessor(
    ds, sequence_length, dataset_name, min_ratio=0.1, bbox_format='y1x1y2x2', class_names=None):
  is_training = sequence_length.get('is_training', True)

  if class_names is not None:
    class_names = tf.constant(data_utils.load_class_names(class_names))

  def filter_fn(ex):
    boxes = tf.reshape(ex['bbox'], [-1, 4])
    area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    ind = area > min_ratio
    return tf.math.reduce_any(ind)

  ds = ds.filter(filter_fn)

  def to_inputs_and_targets(ex):
    img = ex['image']
    if img.dtype == tf.string:
      img = tf.image.decode_jpeg(img, channels=3)

    boxes = tf.reshape(ex['bbox'], [-1, 4])
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)

    img, img_mask, this_image_info = resize_and_pad_default(
      img, is_training, boxes=boxes, box_labels=ex['label'])
    image_info, masks, boxes, labels, indices = this_image_info

    # filter the bbox with min num of pixel.
    area = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    valid_inds = area > np.prod(config.IMAGE_INPUT_SIZE) * min_ratio
    boxes = boxes[valid_inds]
    labels = labels[valid_inds]
    img_mask = tf.cast(img_mask, tf.int32)

    if tf.shape(labels)[0] > 0:
      # randomly sample one labels.
      rand_int = tf.random.uniform(shape=[], maxval= tf.shape(labels)[0], dtype=tf.int32)
      label = labels[rand_int]
      box = boxes[rand_int]
      mask = data_utils.box_mask(tf.cast(box, tf.int32), config.IMAGE_INPUT_SIZE)
      image_input_masks_ori = tf.cast(mask == 0, tf.int32)
      img_mask = img_mask * image_input_masks_ori

      if class_names is not None:
        label = class_names[label]

      region_description = convert_bboxes_to_str(
        tf.expand_dims(box, axis=0),
        tf.expand_dims(label, axis=0),
      )
      valid = True
    else:
      # No valid labels, skip this image
      image_input_masks_ori = tf.ones(tf.shape(img)[:2], tf.int32)
      region_description = "none"
      valid = False

    image_target = tf.image.resize(
      img,
      config.IMAGE_TARGET_SIZE,
      method=tf.image.ResizeMethod.BICUBIC
    )
    image_target_masks = tf.cast(image_input_masks_ori == 0, tf.int32)

    region_description = tf.strings.regex_replace(region_description, r"\\", " ")
    region_description = tf.strings.lower(region_description)

    text_inputs = random_element(Prompt().get_prompt_list('Image_Inpainting', dataset_name))
    text_inputs = valid_regex_replace(text_inputs, {"{}":  region_description})
    text_inputs = tf.strings.join(["[Image] [S] ", text_inputs])

    return {
      'image_inputs': img,
      'image_input_masks': img_mask,
      'image_targets': image_target,
      'image_target_masks': image_target_masks,
      'text_inputs': text_inputs,
      "valid": valid,
    }
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.filter(lambda x: x["valid"])
  return ds


def vqa_preprocessor(ds, sequence_length, flatten_first=True):
  is_training = sequence_length.get('is_training', True)

  @seqio.map_over_dataset(num_seeds=2)
  def to_inputs_and_targets(ex, seeds):
    seeds = list(seeds)
    img = ex["image"]
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img, img_mask, this_image_info = resize_and_pad_default(img, is_training)
    text_inputs = ex['text_inputs']
    text_targets = ex['text_targets']

    prompt_list = random_element(Prompt().get_prompt_list('VQA_short_prompt', ""), seeds[0])
    text_inputs = valid_regex_replace(prompt_list, {"{}": text_inputs})
    text_inputs = tf.strings.join(["[Text] [S] ", text_inputs])

    features = {
      'text_inputs': text_inputs,
      'image_inputs': img,
      'image_input_masks': img_mask,
    }
    meta = {}
    if "id" in ex:
      meta["example_id"] = ex["id"]

    if 'text_targets' in ex:
      if len(text_targets.shape) == 0:
        meta["label"] = text_targets
      else:
        meta['all_references'] = text_targets
      features['text_targets'] = text_targets
    else:
      # Testing example
      features['text_targets'] = ""

    features["meta"] = meta
    return features

  if flatten_first:
    # If there are multiple questions per an image, flatten to get one question per an example
    parts = ["text_inputs"]
    if "id" in ds.element_spec:
      parts.append("id")
    if "text_targets" in ds.element_spec:
      parts.append("text_targets")
    ds = flatten_parts(ds, parts)

  return to_inputs_and_targets(ds)


def box_classification_preprocessor(ds, sequence_length, dataset_name,
                                    class_names=None, bbox_format='y1x1y2x2'):
  is_training = sequence_length.get('is_training', True)
  if class_names is not None:
    class_names = tf.constant(load_class_names(class_names))

  def to_inputs_and_targets(ex):
    img = ex['image']
    if img.dtype == tf.string:
      img = tf.image.decode_jpeg(img, channels=3)
    boxes = tf.reshape(ex['bbox'], [-1, 4])
    if bbox_format == 'x1x2y1y2':
      boxes = tf.stack([boxes[:,2], boxes[:,0], boxes[:,3], boxes[:,1]], axis=1)
    elif bbox_format == 'x1y1x2y2':
      boxes = tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis=1)
    img, img_mask, this_image_info = resize_and_pad_default(
      img, is_training, boxes=boxes, box_labels=ex['label'])
    image_info, masks, boxes, labels, indices = this_image_info

    if tf.shape(boxes)[0] > 0:
      rand_idx = tf.random.uniform([], minval=0, maxval=tf.shape(boxes)[0], dtype=tf.int32)
      boxes = boxes[rand_idx:rand_idx+1]
      label = labels[rand_idx:rand_idx+1]

      if class_names is not None:
        label = class_names[label[0]]
      text_targets = tf.strings.regex_replace(label, r"\\", " ")
      text_targets = tf.strings.lower(text_targets)

      box_str = convert_bboxes_to_str(boxes)
      prompt = random_element(Prompt().get_prompt_list('Box_Classification_Scene', dataset_name))
      text_inputs = valid_regex_replace(prompt, {"{}": tf.strings.lower(box_str)})
      text_inputs = tf.strings.join(["[Text] [S] ", text_inputs])
      valid = True
    else:
      text_inputs = tf.constant("")
      text_targets = tf.constant("")
      valid = False

    features = dict(
      image_inputs=img,
      image_input_masks=img_mask,
      valid=valid,
      text_inputs=text_inputs,
      text_targets=text_targets,
      meta=dict(
        label=text_targets,
      )
    )
    if not is_training and class_names is not None:
      # Test time the model selects from a list of answer options
      features["choices"] = class_names
    return features

  ds = flatten_by_label(ds, ["bbox", "label"])
  ds = ds.map(to_inputs_and_targets)
  ds = ds.filter(lambda x: x["valid"])
  return ds
