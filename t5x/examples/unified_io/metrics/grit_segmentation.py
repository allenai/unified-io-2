from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy import ndimage


# https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py
# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
  """
  Convert binary mask to boundary mask.
  :param mask (numpy array, uint8): binary mask
  :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
  :return: boundary mask (numpy array)
  """
  h, w = mask.shape
  img_diag = np.sqrt(h ** 2 + w ** 2)
  dilation = int(round(dilation_ratio * img_diag))
  if dilation < 1:
    dilation = 1
  # Pad image so mask truncated by the image border is also considered as boundary.
  new_mask = np.pad(mask, [[1, 1], [1, 1]], constant_values=0)
  kernel = np.ones((3, 3), dtype=np.uint8)
  new_mask_erode = ndimage.binary_erosion(new_mask, kernel, iterations=dilation)
  mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
  # G_d intersects G in the paper.
  return np.logical_xor(mask, mask_erode)


def compute_iou(mask1, mask2, verbose=False):
  # resize predicted mask to be the same size as gt mask
  if mask1.shape != mask2.shape:
    raise NotImplementedError()

  intersection = np.sum(np.logical_and(mask1, mask2))
  union = np.sum(np.logical_or(mask1, mask2))

  iou = intersection / (union + 1e-6)

  if verbose:
    return iou, intersection, union

  return iou


def assign_segmentations(pred_masks: list, gt_masks: list):
  n1 = len(pred_masks)
  n2 = len(gt_masks)
  cost = np.zeros([n1,n2])
  ious = np.zeros([n1,n2])
  for i,mask1 in enumerate(pred_masks):
    for j,mask2 in enumerate(gt_masks):
      iou = compute_iou(mask1,mask2)
      ious[i,j] = iou
      cost[i,j] = -iou

  # solve assignment
  pred_mask_ids, gt_mask_ids = linear_sum_assignment(cost)
  pair_ids = list(zip(pred_mask_ids, gt_mask_ids))

  # select assignments with iou > 0
  pair_ids = [(i,j) for i,j in pair_ids if ious[i,j] > 0]
  pairs = [(pred_masks[i],gt_masks[j]) for i,j in pair_ids]
  pair_ious = [ious[i,j] for i,j in pair_ids]

  return pairs, pair_ious, pair_ids


# expects numpy arrays, could change to RLEs and use rle iou / merge fuction above
def seg_metric(pred_masks: list, gt_masks: list, stuff: bool, return_pairs=False) -> float:
  """
  pred_masks: list of numpy arrays representing binary masks
  gt_masks: list of numpy arrays representing binary masks
  stuff: boolean for evaluation type (False for "Thing")
  pairs: return assignment pairs between prediction and gt instances
  """
  if stuff:  # merge masks into to single mask
    pred_masks = [np.logical_or.reduce(pred_masks)] if pred_masks else []
    gt_masks = [np.logical_or.reduce(gt_masks)]

  num_pred = len(pred_masks)
  num_gt = len(gt_masks)
  if num_pred == 0 and num_gt == 0:
    return 1 if not return_pairs else (1, [], [])
  elif min(num_pred,num_gt) == 0 and max(num_pred,num_gt) > 0:
    return 0 if not return_pairs else (0, [], [])

  # uses BoundaryIoU
  # TODO double check mask_to_boundary is officially used, its not clear from the grit codebase
  pred_boundaries = [mask_to_boundary(m) for m in pred_masks]
  gt_boundaries = [mask_to_boundary(m) for m in gt_masks]

  pairs, pair_ious, pair_ids = assign_segmentations(pred_boundaries,gt_boundaries)
  num_detected = len(pairs)
  num_missed = num_gt - num_detected
  score = np.sum(pair_ious) / (num_pred + num_missed)

  return score if not return_pairs else (score, pairs, pair_ids)
