# The object keypoint score (OKS) is computed and averaged within each image
import numpy as np
from scipy.optimize import linear_sum_assignment

N_KEYPOINTS = 17
N_DIM = 3


def get_bbox_from_kp(kp):
  k_array3d = np.reshape(np.array(kp),(N_KEYPOINTS,N_DIM))
  kp = k_array3d[:,:2]
  k_vis = k_array3d[:,2]
  kp_only_labeled = kp[k_vis > 0]
  if len(kp_only_labeled) == 0:
    raise ValueError("All points are marked as not visible!")
  x_min = kp_only_labeled[:,0].min()
  y_min = kp_only_labeled[:,1].min()
  x_max = kp_only_labeled[:,0].max()
  y_max = kp_only_labeled[:,1].max()
  bbox = np.array([x_min, y_min, x_max, y_max])
  return bbox


def computeOks(dts, gts):
  """
  analogous to computing IoUs for localization / segmentation
  """
  ious = np.zeros((len(dts), len(gts)))
  sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
  variances = (sigmas * 2)**2

  # compute oks between each detection and ground truth object
  for j, gt in enumerate(gts):
    g = np.array(gt)
    xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
    x_min, y_min, x_max, y_max = get_bbox_from_kp(gt)
    area = (y_max-y_min)*(x_max-x_min)
    for i, dt in enumerate(dts):
      d = np.array(dt)
      xd = d[0::3]; yd = d[1::3]
      dx = xd - xg
      dy = yd - yg
      e = (dx**2 + dy**2) / variances / (area+np.spacing(1)) / 2
      e = e[vg > 0]
      ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
  return ious


def assign_instances(pred_points: list, gt_points: list):
  ious = computeOks(pred_points, gt_points)
  cost = -ious

  # solve assignment
  pred_ids, gt_ids = linear_sum_assignment(cost)
  pair_ids = list(zip(pred_ids, gt_ids))

  # select assignments with iou > 0
  pair_ids = [(i,j) for i,j in pair_ids if ious[i,j] > 0]
  pairs = [(pred_points[i],gt_points[j]) for i,j in pair_ids]
  pair_ious = [ious[i,j] for i,j in pair_ids]

  return pairs, pair_ious, pair_ids


def kp_metric(pred_points: list, gt_points: list, return_pairs=False) -> float:
  num_pred = len(pred_points)
  num_gt = len(gt_points)
  if num_pred == 0 and num_gt == 0:
    return 1 if not return_pairs else (1, [], [])
  elif min(num_pred,num_gt) == 0 and max(num_pred,num_gt) > 0:
    return 0 if not return_pairs else (0, [], [])

  pairs, pair_ious, pair_ids = assign_instances(pred_points, gt_points)
  num_detected = len(pairs)
  num_missed = num_gt - num_detected
  score = np.sum(pair_ious) / (num_pred + num_missed)

  return score if not return_pairs else (score, pairs, pair_ids)


def kp_metric_wrapper(prediction, ground_truth, return_pairs=False):
  if 'output' in prediction:
    prediction = prediction['output']
  assert ground_truth['output']['example_id'] == prediction['example_id']
  pred_points = prediction['points']
  gt_points = ground_truth['output']['points']
  return kp_metric(pred_points, gt_points, return_pairs)