import pdb
from typing import List, Tuple

import numpy as np
import re
import tensorflow as tf

from t5x.examples.unified_io import config
from t5x.examples.unified_io.config import VOCAB_START, NUM_DETECTION_BIN
from t5x.examples.unified_io.data.data_utils import get_default_vocabulary
from transforms3d import euler

VOCAB = get_default_vocabulary()


bbox_regex = re.compile(" ?".join(r"<extra_id_([0-9]+)>" for _ in range(4)) + " ?([^<]+)")
bbox_localization_regex = re.compile(" ?".join(r"<extra_id_([0-9]+)>" for _ in range(4)))
point_regex = re.compile(" ?".join(r"<extra_id_([0-9]+)>" for _ in range(2)) + " ?([^<]+)")
xyz_regex = re.compile(" ?".join(r"<extra_id_([0-9]+)>" for _ in range(3)))


def extract_bboxes_from_text(text: str, image_size=None, num_bin=NUM_DETECTION_BIN, vocab_start=VOCAB_START, use_label=True) -> Tuple[np.ndarray, List[str]]:
  """Extract boxes mentioned in `text` using our location encoding"""
  if text.endswith(">"):  # match the re if no label
    text = text + " "
  return extract_coordinates_from_text(text, image_size, num_bin, vocab_start, 4, use_label=use_label)


def extract_points_from_text(text, image_size=None, num_bin=NUM_DETECTION_BIN, vocab_start=VOCAB_START) -> Tuple[np.ndarray, List[str]]:
  """Extract points mentioned in `text` using our location encoding"""
  return extract_coordinates_from_text(text, image_size, num_bin, vocab_start, 2)


def extract_xyz_from_text(text, image_size=None, num_bin=NUM_DETECTION_BIN, vocab_start=VOCAB_START) -> Tuple[np.ndarray, List[str]]:
  """Extract 3-tuple coordinates mentioned in `text` using our location encoding"""
  return extract_coordinates_from_text(text, image_size, num_bin, vocab_start, 3, use_label=False)


def extract_coordinates_from_text(text, image_size=None, num_bin=NUM_DETECTION_BIN,
                                  vocab_start=VOCAB_START, n_coordinates=2, allow_all=False,
                                  use_label=True):
  boxes = []
  class_names = []
  if n_coordinates == 2:
    exp = point_regex
  elif n_coordinates == 4 and use_label:
    exp = bbox_regex
  elif n_coordinates == 4:
    exp = bbox_localization_regex
  elif n_coordinates == 3:
    exp = xyz_regex
  else:
    raise NotImplementedError()
  
  for match in exp.finditer(text):
    token_ids = [int(match.group(i)) for i in range(1, 1+n_coordinates)]
    if not allow_all and not all(vocab_start <= ix < (vocab_start + num_bin) for ix in token_ids):
      # Contains non-location token ids
      continue
    if use_label:
      class_names.append(match.group(n_coordinates+1).strip())
    boxes.append(token_ids)

  if not boxes:
    return np.zeros((0, n_coordinates), dtype=np.int64), []
  boxes = (np.array(boxes) - vocab_start) / num_bin
  if image_size is not None:
    assert n_coordinates % 2 == 0
    h, w = image_size[:2]
    factor = [h, w] * (n_coordinates//2)
    boxes = boxes * np.expand_dims(np.array(factor), 0)

  return boxes, class_names


def build_target_image(image, image_info, gray_scale=False,
                       resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, to_int=False,
                       resize_first=False):
  """Build an image from `image` that is the same size as the original input image, before
  any scaling and cropping was done."""
  assert isinstance(gray_scale, bool)

  image = (image + 1) / 2.0  # Undo pre-processings
  image = tf.clip_by_value(image, 0, 1)  # We can (rarely) get negative pixel values, clip them here
  if gray_scale:
    image = tf.reduce_mean(image, -1, keepdims=True)

  off_x = int(image_info[7])
  off_y = int(image_info[8])
  if not (off_x == 0 and off_y == 0):
    raise NotImplementedError()

  src_h = int(image_info[3])
  src_w = int(image_info[4])

  if config.IMAGE_INPUT_SIZE != config.IMAGE_TARGET_SIZE:
    # TODO can we support this? Its a bit tricky since the padding
    # value are given w.r.t. input image not target imaget5x/examples/unified_io/data/grit_tasks.py
    resize_first = True

  if not resize_first:
    top_pad = int(image_info[0])
    left_pad = int(image_info[1])
    scaled_height = int(image_info[-2])
    scaled_width = int(image_info[-1])
    assert top_pad == 0 or left_pad == 0
    image = image[top_pad:top_pad+scaled_height, left_pad:left_pad+scaled_width]

    src_h = int(image_info[3])
    src_w = int(image_info[4])
    image = tf.image.resize(image, [src_h, src_w], method=resize_method)
  else:
    # Resize then crop
    w = max(src_h, src_w)
    image = tf.image.resize(image, [w, w], method=resize_method)
    if config.PAD_ONE_SIDE:
      image = image[:src_h, :src_w]
    else:
      if src_h > src_w:
        delta = (src_h - src_w) // 2
        image = image[:, delta:delta+src_w]
      else:
        delta = (src_w - src_h) // 2
        image = image[delta:delta+src_h, :]

  if to_int:
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
  return image.numpy()


def build_depth_prediction(image, image_info, max_depth,
                           resize_method=tf.image.ResizeMethod.BILINEAR):
  # TODO maybe there is a better resize method for depth?
  image = build_target_image(image, image_info, gray_scale=True, resize_method=resize_method)
  return image * max_depth


def extract_actions_from_prediction(text, num_bin=NUM_DETECTION_BIN, vocab_start=VOCAB_START):
  """Extract actions as used in vima"""
  action_pattern = re.compile(
    r'(\d+): ((push|pick) (from)) \( <extra_id_(\d+)> <extra_id_(\d+)> <extra_id_(\d+)> \) (and) ((push|place) (to)) \( <extra_id_(\d+)> <extra_id_(\d+)> <extra_id_(\d+)> \)'
  )
  actions = None
  step_indices = None
  for match in action_pattern.finditer(text):
    if actions is None:
      # to return null if no match
      actions, step_indices = [], []
    step_idx = match.group(1)
    step_indices.append(step_idx)
    pose0_primitive = match.group(2)
    ee_type = 0.0 if pose0_primitive == "pick from" else 1.0
    pose0 = tuple(int(num) for num in match.group(5, 6, 7))
    pose1 = tuple(int(num) for num in match.group(12, 13, 14))
    # back to 0-1
    pose0 = (np.array(pose0) - vocab_start) / num_bin
    pose1 = (np.array(pose1) - vocab_start) / num_bin
    x0, y0, rot0 = pose0
    x1, y1, rot1 = pose1
    # reconstruct action in dataset format
    actions.append((x0, y0, rot0 * 360.0, x1, y1, rot1 * 360.0, ee_type))
  return actions, step_indices


def unnormalize_pos(action, action_position_bounds):
  action["pose0_position"] = (
      action["pose0_position"]
      * (action_position_bounds["high"] - action_position_bounds["low"])
      + action_position_bounds["low"]
  )
  action["pose1_position"] = (
      action["pose1_position"]
      * (action_position_bounds["high"] - action_position_bounds["low"])
      + action_position_bounds["low"]
  )
  assert np.all(action["pose0_position"] >= action_position_bounds["low"]) and np.all(
    action["pose0_position"] <= action_position_bounds["high"]
  ), action
  assert np.all(action["pose1_position"] >= action_position_bounds["low"]) and np.all(
    action["pose1_position"] <= action_position_bounds["high"]
  ), action
  return action


def eulerXYZ_to_quatXYZW(rotation):
  euler_zxy = (rotation[2], rotation[0], rotation[1])
  quaternion_wxyz = euler.euler2quat(*euler_zxy, axes="szxy")
  q = quaternion_wxyz
  quaternion_xyzw = (q[1], q[2], q[3], q[0])
  return quaternion_xyzw


def reconstruct_vima_action(actions):
  VIMA_ACTION_BOUNDS = {
    "low": np.array([0.25, -0.5], dtype=np.float32),
    "high": np.array([0.75, 0.50], dtype=np.float32),
  }

  action_dict = {
    "pose0_position": [],
    "pose0_rotation": [],
    "pose1_position": [],
    "pose1_rotation": [],
  }
  for action in actions:
    pos0_x, pos0_y, pos0_rot_z, pos1_x, pos1_y, pos1_rot_z, ee = action
    action_dict["pose0_position"].append([pos0_x, pos0_y])
    action_dict["pose0_rotation"].append(
      eulerXYZ_to_quatXYZW((0.0, 0.0, pos0_rot_z))
    )
    action_dict["pose1_position"].append([pos1_x, pos1_y])
    action_dict["pose1_rotation"].append(
      eulerXYZ_to_quatXYZW((0.0, 0.0, pos1_rot_z))
    )
  for k in action_dict.keys():
    action_dict[k] = np.asarray(action_dict[k], dtype=np.float32)
  action_dict = unnormalize_pos(action_dict, VIMA_ACTION_BOUNDS)
  return action_dict


def undo_box_preprocessing(boxes, image_info):
  """"Convert boxes relative to the scaled/cropped input image to pixel
  coordinates relative to the original image"""
  # Undo padding
  top_pad, left_pad = image_info[0], image_info[1]
  paddings = np.array([top_pad, left_pad, top_pad, left_pad], dtype=boxes.dtype)
  boxes = boxes - paddings

  # Not sure how to handle offsets at the moment (simple addition?)
  # for now just require them to be zero as should be the case during eval
  off_y = int(image_info[7])
  off_x = int(image_info[8])
  assert off_x == off_y == 0

  # Undo the scaling
  inv_scale = image_info[2]
  boxes = boxes * inv_scale

  # clip in case the model predicted a region in the padded area
  h, w = image_info[3:5]
  boxes = np.maximum(boxes, 0)
  boxes = np.minimum(boxes, [h, w, h, w])
  return boxes
