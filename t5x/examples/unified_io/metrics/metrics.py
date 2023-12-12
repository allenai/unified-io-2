"""Metrics for UiO2 tasks"""
from collections import Counter

from absl import logging
from typing import List, Sequence

import numpy as np
from seqio.metrics import Scalar, Text

from t5x.examples.unified_io.metrics.utils import extract_coordinates_from_text, \
  undo_box_preprocessing

from t5x.examples.unified_io.evaluator import UnifiedIOOutput

from t5x.examples.unified_io.metrics.grit_localization import compute_iou
from t5x.examples.unified_io.metrics.grit_vqa import preprocess_answer as vqa_preprocessing
from t5x.examples.unified_io import config


def exact_match(targets, predictions: List[UnifiedIOOutput],
                aux_values, print_examples=True):
  if isinstance(targets[0], np.ndarray):
    # Multiple correct answers stored in a numpy object array
    matches = [pred.text.lower() in [x.decode("utf-8").lower() for x in target] for target, pred in zip(targets, predictions)]
  else:
    if isinstance(targets[0], bytes):
      targets = [x.decode("utf-8") for x in targets]
    matches = [pred.text.lower() == target.lower() for target, pred in zip(targets, predictions)]
  if print_examples:
    ixs = np.random.choice(len(targets), min(20, len(targets)), replace=False)
    examples = [f"pred={predictions[i].text} gt={targets[i]}" for i in ixs]
    for ex in examples:
      logging.info(ex)
  return {
    "score": np.mean(matches),
  }


def vqa_score(target, pred):
  pred = vqa_preprocessing(pred)
  if isinstance(target, list):
    target = Counter(vqa_preprocessing(x) for x in target)
    return min(target[pred] / 3.0, 1)
  else:
    return float(vqa_preprocessing(pred) == vqa_preprocessing(target))


def vqa_metric(targets: Sequence, predictions: Sequence[UnifiedIOOutput], aux_values):
  if isinstance(targets[0], np.ndarray):
    targets = [[ans.decode("utf-8") for ans in answer_set] for answer_set in targets]
  else:
    targets = [answer.decode("utf-8") for answer in targets]
  score = np.mean([vqa_score(t, p.text) for t, p in zip(targets, predictions)])
  n_targets = len(targets)
  ixs = np.random.choice(n_targets, min(n_targets, 20), replace=False)
  examples = [f"{predictions[i].text.lower()} (gt(s)={', '.join(targets[i])})" for i in ixs]
  return {
    "score": Scalar(score),
    "examples": Text(",  ".join(examples))
  }


def ref_exp_metric(targets, predictions: List[UnifiedIOOutput],
                   aux_values, original_scale=True):
  total_acc = 0
  total_iou = 0
  for target, pred in zip(targets, predictions):
    gt_boxes, image_info, src_boxes = target["boxes"], target["image_info"], target["src_boxes"]
    if len(gt_boxes) != 1:
      raise ValueError("Should always be one ground truth box")

    p_boxes, classes = extract_coordinates_from_text(
      pred.text, image_size=config.IMAGE_INPUT_SIZE, n_coordinates=4, use_label=False)
    if original_scale:
      p_boxes = undo_box_preprocessing(p_boxes, image_info)
      h, w = image_info[3:5]
      gt_boxes = src_boxes * np.array([h, w, h, w]).reshape(1, 4)
    if len(p_boxes) == 0:
      iou = 0
    else:
      iou = compute_iou(p_boxes[0], gt_boxes[0])
    total_iou += iou
    total_acc += float((iou > 0.5))

  n = len(predictions)
  return dict(acc=total_acc/n, iou=total_iou/n)


def null_metric(targets, predictions, aux_values):
  return {}
