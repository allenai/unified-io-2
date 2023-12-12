from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(bbox1: list, bbox2: list, verbose: bool=False):
  x1, y1, x2, y2 = bbox1
  x1_, y1_, x2_, y2_ = bbox2

  x1_in = max(x1, x1_)
  y1_in = max(y1, y1_)
  x2_in = min(x2, x2_)
  y2_in = min(y2, y2_)

  intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)
  area1 = compute_area(bbox1, invalid=0)
  area2 = compute_area(bbox2, invalid=0)
  union = area1 + area2 - intersection
  iou = intersection / (union + 1e-6)

  if verbose:
    return iou, intersection, union

  return iou


def compute_area(bbox: list, invalid: float=None) -> float:
  x1, y1, x2, y2 = bbox

  if (x2 <= x1) or (y2 <= y1):
    area = invalid
  else:
    area = (x2 - x1) * (y2 - y1)

  return area


def assign_boxes(pred_boxes: List[List], gt_boxes: List[List]):
  n1 = len(pred_boxes)
  n2 = len(gt_boxes)
  cost = np.zeros([n1,n2])
  ious = np.zeros([n1,n2])
  for i,bbox1 in enumerate(pred_boxes):
    for j,bbox2 in enumerate(gt_boxes):
      iou = compute_iou(bbox1,bbox2)
      ious[i,j] = iou
      cost[i,j] = 1-iou

  # solve assignment
  pred_box_ids, gt_box_ids = linear_sum_assignment(cost)
  pair_ids = list(zip(pred_box_ids, gt_box_ids))

  # select assignments with iou > 0
  pair_ids = [(i,j) for i,j in pair_ids if ious[i,j] > 0]
  pairs = [(pred_boxes[i],gt_boxes[j]) for i,j in pair_ids]
  pair_ious = [ious[i,j] for i,j in pair_ids]

  return pairs, pair_ious, pair_ids


def loc_metric(pred_boxes: List[List], gt_boxes: List[List]) -> float:
  num_pred = len(pred_boxes)
  num_gt = len(gt_boxes)
  if num_pred == 0 and num_gt == 0:
    return 1
  elif min(num_pred,num_gt) == 0 and max(num_pred,num_gt) > 0:
    return 0

  pairs, pair_ious, pair_ids = assign_boxes(pred_boxes,gt_boxes)
  num_detected = len(pairs)
  num_missed = num_gt - num_detected
  return np.sum(pair_ious) / (num_pred + num_missed)