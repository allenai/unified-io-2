
import numpy as np
import os
from scipy.spatial.transform import Rotation
import math
import time

# https://github.com/princeton-vl/oasis/blob/master/eval/absolute_surface_normal/eval_abs_normal.py
def ang_error(pred_normal, gt_normal, ROI=None):
  '''
  Inputs
      pred_normal: A numpy array of HxWx3, float32
      gt_normal:   A numpy array of HxWx3, float32
      ROI:         A numpy array of HxW,   uint8.
                   If None, the entire image is ROI
  Return
      The angular differences between pred and gt in the ROI
  '''
  assert(pred_normal.shape[0] == gt_normal.shape[0])
  assert(gt_normal.shape[0] == gt_normal.shape[0])
  assert(len(pred_normal.shape) == 3 and len(gt_normal.shape) == 3)

  # normalize
  pred_normal = pred_normal / \
                np.linalg.norm(pred_normal, ord=2, axis=2, keepdims=True)  # HxWx3
  gt_normal = gt_normal / \
              np.linalg.norm(gt_normal, ord=2, axis=2, keepdims=True)		# HxWx3

  # calculate the angle difference
  dot_prod = np.multiply(pred_normal, gt_normal)				# HxWx3
  dot_prod = np.sum(dot_prod, axis=2)						    # HxW
  dot_prod = np.clip(dot_prod, a_min=-1.0, a_max=1.0)

  angles = np.arccos(dot_prod)
  angles = np.degrees(angles)

  if ROI is None:
    # assert(np.all(angles <= 180.))
    return angles.flatten()
  else:
    # assert(np.all(angles[ROI > 0] <= 180.))
    return angles[ROI > 0]


def evaluate_normal(pred_normals, gt_normals, ROIs=None, verbose=False):
  '''
  Inputs
      pred_normal: A list of numpy arrays of HxWx3
      gt_normal:   A list of numpy arrays of HxWx3
      ROIs:		 A list of numpy arrays of HxW.
  Return
      mean_err:    The mean angular difference between the predicted and ground-truth normals. Measured in degree.
      median_err:  The median angular difference between the predicted and ground-truth normals. Measured in degree.
      below_11_25: The percentage of pixels whose angular difference is less than 11.25 degree.
      below_22_5:  The percentage of pixels whose angular difference is less than 22.50 degree.
      below_30:    The percentage of pixels whose angular difference is less than 30.00 degree.
  '''

  assert(len(pred_normals) == len(gt_normals))

  if ROIs is None:
    ROIs = [None for i in range(len(gt_normals))]

  angles = []
  for pred, gt, ROI in zip(pred_normals, gt_normals, ROIs):
    _angles = ang_error(pred, gt, ROI)
    angles.append(_angles)

  angles = np.concatenate(angles)
  n_pixels = float(len(angles))

  mean_err = np.mean(angles)
  median_err = np.median(angles)
  below_11_25 = float(np.sum(angles < 11.25)) / n_pixels * 100
  below_22_5 = float(np.sum(angles < 22.5)) / n_pixels * 100
  below_30    = float(np.sum(angles < 30))    / n_pixels * 100

  if verbose:
    print(f"Mean Error:      {mean_err:.2f}")
    print(f"Median Error:    {median_err:.2f}")
    print(f"Inliers < 11.25°:  {below_11_25:.0f}%")
    print(f"Inliers < 22.5°:   {below_22_5:.0f}%")
    print(f"Inliers < 30°:     {below_30:.0f}%")

  return mean_err, median_err, below_11_25, below_22_5, below_30

def evaluate_normal_fast(U: np.ndarray, V: np.ndarray, M: np.ndarray):
  h,w = M.shape
  M_flat = np.argwhere(M.reshape(h*w)).squeeze()
  U_flat = U.reshape(h*w,3)[M_flat]
  V_flat = V.reshape(h*w,3)[M_flat]
  inliers = num_inliers(V_flat, U_flat)
  return 100 * inliers/U_flat.shape[0]

def rgb2normal(rgb):
  n = (rgb-122.5)/122.5
  n /= np.linalg.norm(n, ord=2, axis=2, keepdims=True)
  return n

def normal2rgb(n):
  rgb = (n + 1.0)/2.0
  rgb *= 255.0
  return rgb.astype(np.uint8)

def rotate_with_mask(X, M, R):
  out = X @ R.T
  out[~M] = X[~M]
  return out

def get_rotation_matrix(U,V,M):
  U = U.copy()
  V = V.copy()
  M = M.copy()
  # mask out invalid vectors
  h,w = M.shape
  M = np.argwhere(M.reshape(h*w)).squeeze()
  U = U.reshape(h*w,3)[M]
  V = V.reshape(h*w,3)[M]
  # estimation
  R_hat, _ = Rotation.align_vectors(V,U)
  return R_hat.as_matrix()

#https://stackoverflow.com/questions/15707056/get-time-of-execution-of-a-block-of-code-in-python-2-7
#https://stackoverflow.com/questions/739654/how-to-make-function-decorators-and-chain-them-together
def time_this(func):
  def wrapper(*args, **kwargs):
    beg_ts = time.time()
    retval = func(*args, **kwargs)
    end_ts = time.time()
    time_ts = end_ts - beg_ts
    return retval, time_ts
  return wrapper

def get_rotation_matrix_RANSAC(U: np.ndarray, V: np.ndarray, M: np.ndarray, n_iters=500, verbose=False, mode=None, r_size=500, calc_score=False) -> np.ndarray:
  """
  Finds a rotation matrix to align normals from U to V using RANSAC
  The values of the inputs U and V should be between -1 and 1 (normalized vectors)
  Args:
      U (ndarray): ndarray of shape (H, W, 3) and dtype float.
      V (ndarray): ndarray of shape (H, W, 3) and dtype float.
      M (ndarray): A List of tuple where,
      n_iters (int): Integer denoting the number of iterations in RANSAC.
      verbose (bool): Boolean flag to print the intermediate scores everytime a better estimation is found
      mode (str): String specifying the RANSAC variant; one of ['R-RANSAC','...']
  Returns:
      R (ndarray[3, 3]): Best 3D Rotation matrix found by RANSAC.
  """
  if not isinstance(U, np.ndarray):
    raise TypeError(f"U must be a np.ndarray, got {type(U)}")
  elif not isinstance(V, np.ndarray):
    raise TypeError(f"V must be a np.ndarray, got {type(V)}")
  elif U.shape != V.shape:
    raise TypeError(f"U and V must have the same shape, got {U.shape} != {V.shape}")
  elif U.shape[2] == M.shape:
    raise TypeError(f"Mask M must have the same first dimensions of U, got {U.shape[2] != M.shape}")

  h,w = M.shape
  M_flat = np.argwhere(M.reshape(h*w)).squeeze()
  U_flat = U.copy().reshape(h*w,3)[M_flat]
  U_flat /= np.linalg.norm(U_flat, ord=2, axis=1, keepdims=True)
  V_flat = V.copy().reshape(h*w,3)[M_flat]
  V_flat /= np.linalg.norm(V_flat, ord=2, axis=1, keepdims=True)

  # initialize RANSAC
  np.random.seed(0)
  best_inliers = 0
  R_hat = Rotation.from_matrix(np.eye(3))
  rng = np.random.default_rng()

  ali_time = np.zeros(n_iters)
  rot_time = np.zeros(n_iters)
  inl_time = np.zeros(n_iters)
  it_time = np.zeros(n_iters)
  choice_time = np.zeros(n_iters)
  tot_time1 = time.time()

  for i in range(n_iters):
    it_time1 = time.time()
    if i > 0:
      # choose two pairs of normal vectors and solve for rotation
      ind = np.random.choice(U.shape[0], size=2, replace=False)
      # Rotation.align_vectors(a,b) gives you R that transforms b to a.
      (R_hat, _), ali_time[i] = time_this(Rotation.align_vectors)(V_flat[ind,:],U_flat[ind,:])


    # compute # of inliers, and save rotation if best
    if mode=="R-RANSAC":
      # consesus_set_idx, choice_time[i] = time_this(np.random.choice)(U_flat.shape[0], size=500, replace=False)
      # https://stackoverflow.com/questions/8505651/non-repetitive-random-number-in-numpy --> 10x speedup
      consesus_set_idx, choice_time[i] = time_this(rng.choice)(U_flat.shape[0], size=r_size, replace=False)
      V_hat_flat, rot_time[i] = time_this(R_hat.apply)(U_flat[consesus_set_idx])
      inliers, inl_time[i] = time_this(num_inliers)(V_hat_flat, V_flat[consesus_set_idx])
    else:
      V_hat_flat, rot_time[i] = time_this(R_hat.apply)(U_flat)
      inliers, inl_time[i] = time_this(num_inliers)(V_hat_flat, V_flat)

    if inliers > best_inliers:
      best_inliers = inliers
      best_R_hat = R_hat
      if verbose:
        print(f"Iteration {i}: {best_inliers/U_flat.shape[0]}")

    it_time2 = time.time()
    it_time[i] = (it_time2 - it_time1)

  tot_time2 = time.time()
  tot_time = tot_time2 - tot_time1
  if verbose:
    print(f"Average time to align vectors {np.array(ali_time).mean()*1000:.2f}ms")
    print(f"Average time to rotate vectors {np.array(rot_time).mean()*1000:.2f}ms")
    print(f"Average time to calc inliers {np.array(inl_time).mean()*1000:.2f}ms")
    print(f"Average time for choice {np.array(choice_time).mean()*1000:.2f}ms")
    print(f"Average time for iteration {np.array(it_time).mean()*1000:.2f}ms")
    print(f"Total time {n_iters} iterations {tot_time:.2f}s")

  # optionally calculate score
  if calc_score:
    V_hat_flat = best_R_hat.apply(U_flat)
    inliers = num_inliers(V_hat_flat, V_flat)
    score = 100 * inliers/U_flat.shape[0]
    tot_time3 = time.time()
    tot_time = tot_time3 - tot_time1
    if verbose:
      print(f"Total time with score {tot_time:.2f}s")
    return best_R_hat.as_matrix(), score
  else:
    return best_R_hat.as_matrix()

def num_inliers(V_hat_flat, V_flat):
  dot_prod = np.multiply(V_hat_flat, V_flat).sum(axis=1).clip(-1.0,1.0)
  angles = np.degrees(np.arccos(dot_prod))
  inliers = np.sum(angles < 11.25)
  return inliers

def sharp_normal_mask(gt_normals_rgb, angle_threshold=40):
  gt_normals = rgb2normal(gt_normals_rgb) # (H, W, 3)

  normal_right = gt_normals[:,1:,:]  # (H, W-1, 3)
  normal_left =  gt_normals[:,:-1,:] # (H, W-1, 3)

  angle_diff_lr = np.arccos(np.minimum(1,(normal_right * normal_left).sum(2))) / math.pi * 180   # (H, W-1)

  normal_down = gt_normals[1:,:,:]  # (H-1, W, 3)
  normal_up =  gt_normals[:-1,:,:] # (H-1, W, 3)

  angle_diff_ud = np.arccos(np.minimum(1,(normal_down * normal_up).sum(2))) / math.pi * 180   # (H,-1 W)

  mask_lr = angle_diff_lr > angle_threshold # mask
  mask_ud = angle_diff_ud > angle_threshold # mask

  up_down_mask = np.zeros(gt_normals.shape[:2])
  up_down_mask[1:,:] = mask_ud
  up_down_mask[:-1,:] = up_down_mask[:-1,:] + mask_ud

  left_right_mask = np.zeros(gt_normals.shape[:2])
  left_right_mask[:,1:] = mask_lr
  left_right_mask[:,:-1] = left_right_mask[:,:-1] + mask_lr

  invalid_mask = (up_down_mask > 0) | (left_right_mask > 0)

  return invalid_mask

def get_mask_from_normals(normals_rgb):
  # masks out values of [0,0,0], [128,128,128], or [201,201,201]
  valid_mask_0 = (normals_rgb != 0).sum(axis=2) != 0
  valid_mask_128 = (normals_rgb != 128).sum(axis=2) != 0
  valid_mask_201 = (normals_rgb != 201).sum(axis=2) != 0
  valid_mask_sharp = ~sharp_normal_mask(normals_rgb)
  valid_mask = np.logical_and.reduce((valid_mask_0, valid_mask_128, valid_mask_201, valid_mask_sharp))
  return valid_mask

def get_image_and_normals(cfg, task, print_source=True):
  # Import here so this dependency is optional
  from skimage.io import imread

  source = task['meta']['data_source']
  if print_source:
    print(source)
  img_path = os.path.join(cfg.datapaths.images.dir, task['output']['image_id'])
  img = imread(img_path)[:,:,:3]
  normal_path = os.path.join(cfg.datapaths.images.dir, task['output']['out_image_name'])
  normals_rgb = imread(normal_path)[:,:,:3]
  valid_mask = get_mask_from_normals(normals_rgb)
  return img, normals_rgb, valid_mask

def sn_metric(predicted_normals_rgb, gt_normals_rgb, valid_mask, verbose=False, rotate=True, ransac=True, mode="R-RANSAC", r_size=500):
  """
  Produces a score for the predicted normals when compared to the ground truth normals
  The inputs pred and gt are RGB images of surface normals of the inputs (0,255)
  Args:
      pred (ndarray): RGB image of shape (H, W, 3) and dtype uint8.
      gt (ndarray): RGB image of shape (H, W, 3) and dtype uint8.
      verbose (bool): print out intermediate outputs
      rotate (bool): solve for a rotation between pred and gt before evaluation
      ransac (bool): when True, use RANSAC estimation, False use optimal (sensitive to outliers)
  Returns:
      score (float): percentage of normals within 11.25° threshold
  """
  assert predicted_normals_rgb.shape == gt_normals_rgb.shape
  # if pred.shape != gt.shape:   # Optional reshaping
  #     pred = resize(pred, gt.shape, preserve_range=True).astype(np.uint8)
  N_R = rgb2normal(predicted_normals_rgb)
  N = rgb2normal(gt_normals_rgb)
  M = valid_mask
  if rotate and ransac:
    R_hat, score = get_rotation_matrix_RANSAC(N_R,N,M, mode=mode, r_size=r_size, calc_score=True)
    N_estimated = rotate_with_mask(N_R, M, R_hat)
  elif rotate:
    R_hat = get_rotation_matrix(N_R,N,M)
    N_estimated = rotate_with_mask(N_R, M, R_hat)
  else:
    N_estimated = N_R

  if M.sum() == 0:
    if verbose:
      print("no normal labels in ground truth")
    return 1.0
  elif not ransac:
    # score = evaluate_normal([N_estimated], [N], [M], verbose)[2] # below 11.25 threshold
    score = evaluate_normal_fast(N_estimated, N, M)
  return score / 100.0
