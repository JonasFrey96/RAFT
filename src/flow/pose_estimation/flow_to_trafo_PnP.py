import torch
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import cv2
from .pose_estimate_violations import Violation


def filter_pcd(pcd, tol=0.05):
  """
  input:
      pcd : Nx3 torch.float32
  returns:
      mask : N torch.bool
  """
  return pcd[:, 2] > tol


def rvec_tvec_to_H(r_vec, t_vec):
  """
  input:
      r_vec: 3 torch.float32
      t_vec: 3 torch.float32
  returns:
      h: np.array( [4,4] )
  """
  rot = R.from_rotvec(r_vec)
  h = np.eye(4)
  h[:3, :3] = rot.as_matrix()
  h[:3, 3] = t_vec.T
  return h


def get_H(pcd):
  pcd_ret = torch.ones(
    (pcd.shape[0], pcd.shape[1] + 1), device=pcd.device, dtype=pcd.dtype
  )
  pcd_ret[:, :3] = pcd
  return pcd_ret


def flow_to_trafo_PnP(*args, **kwargs):
  """
  input:
    real_br: torch.tensor torch.Size([2])
    real_tl: torch.tensor torch.Size([2])
    ren_br: torch.tensor torch.Size([2])
    ren_tl: torch.tensor torch.Size([2])
    flow_mask: torch.Size([480, 640])
    u_map: torch.Size([480, 640])
    v_map: torch.Size([480, 640])
    K_ren: torch.Size([3, 3])
    render_d: torch.Size([480, 640])
    h_render: torch.Size([4, 4])
    h_real_est: torch.Size([4, 4])
  output:
    suc: bool
    h:  torch.Size([4, 4])
  """
  real_br = kwargs["real_br"]
  real_tl = kwargs["real_tl"]
  ren_br = kwargs["ren_br"]
  ren_tl = kwargs["ren_tl"]
  flow_mask = kwargs["flow_mask"]
  u_map = kwargs["u_map"]
  v_map = kwargs["v_map"]
  K_ren = kwargs["K_ren"]
  K_real = kwargs["K_real"]
  render_d = kwargs["render_d"]
  h_render = kwargs["h_render"]
  h_real_est = kwargs["h_real_est"]

  typ = u_map.dtype

  # Grid for upsampled real
  grid_real_h = torch.linspace(
    int(real_tl[0]), int(real_br[0]), 480, device=u_map.device
  )[:, None].repeat(1, 640)
  grid_real_w = torch.linspace(
    int(real_tl[1]), int(real_br[1]), 640, device=u_map.device
  )[None, :].repeat(480, 1)
  # Project depth map to the pointcloud real
  cam_scale = 10000
  real_pixels = torch.stack(
    [
      grid_real_w[flow_mask],
      grid_real_h[flow_mask],
      torch.ones(grid_real_h.shape, device=u_map.device, dtype=u_map.dtype)[flow_mask],
    ],
    dim=1,
  ).type(typ)

  grid_ren_h = torch.linspace(int(ren_tl[0]), int(ren_br[0]), 480, device=u_map.device)[
    :, None
  ].repeat(1, 640)
  grid_ren_w = torch.linspace(int(ren_tl[1]), int(ren_br[1]), 640, device=u_map.device)[
    None, :
  ].repeat(480, 1)
  crop_d_pixels = torch.stack(
    [
      grid_ren_w.flatten(),
      grid_ren_h.flatten(),
      torch.ones(grid_ren_w.shape, device=u_map.device, dtype=torch.float32).flatten(),
    ],
    dim=1,
  ).type(typ)
  K_inv = torch.inverse(K_ren.type(torch.float32)).type(typ)
  P_crop_d = K_inv @ crop_d_pixels.T.type(typ)
  P_crop_d = P_crop_d.type(torch.float32) * render_d.flatten() / cam_scale
  P_crop_d = P_crop_d.T

  render_d_ind_h = torch.linspace(0, 479, 480, device=u_map.device)[:, None].repeat(
    1, 640
  )
  render_d_ind_w = torch.linspace(0, 639, 640, device=u_map.device)[None, :].repeat(
    480, 1
  )
  render_d_ind_h = torch.clamp(
    (render_d_ind_h - u_map).type(torch.float32), 0, 479
  ).type(torch.long)[flow_mask]
  render_d_ind_w = torch.clamp(
    (render_d_ind_w - v_map).type(torch.float32), 0, 639
  ).type(torch.long)[flow_mask]
  if render_d_ind_h.shape[0] < 50:
    return (
      False,
      torch.eye(4, dtype=u_map.dtype, device=u_map.device),
      np.inf,
      0,
      Violation.MINIMAL_NR_VALID_CONSTRAINT,
    )
  # Avoid two different 3D points pointing to the same 2D pixels
  res, indices = np.unique(
    torch.stack([render_d_ind_h, render_d_ind_w]).numpy(), axis=1, return_index=True
  )
  indices = torch.from_numpy(indices)
  render_d_ind_h = render_d_ind_h[indices]
  render_d_ind_w = render_d_ind_w[indices]
  real_pixels = real_pixels[indices]

  render_pixels = torch.stack(
    [render_d_ind_h, render_d_ind_w, torch.ones_like(render_d_ind_w)], dim=1
  )

  # Hacky indexing along two dimensions
  index = render_d_ind_h * 640 + render_d_ind_w

  P_crop_d = P_crop_d[index]

  m = filter_pcd(P_crop_d)

  if torch.sum(m) < 50:
    return (
      False,
      torch.eye(4, dtype=u_map.dtype, device=u_map.device),
      np.inf,
      0,
      Violation.MINIMAL_NR_VALID_CONSTRAINT,
    )
  P_crop_d = P_crop_d[m]
  real_pixels = real_pixels[m]
  render_pixels = render_pixels[m]
  P_ren = P_crop_d

  if kwargs.get("shuffel", "random") == "random":
    # random shuffel
    pts_trafo = min(P_ren.shape[0], kwargs.get("max_corrospondences", 200000))
    idx = torch.randperm(P_ren.shape[0])[0:pts_trafo]

    P_ren = P_ren[idx]
    real_pixels = real_pixels[idx]
    render_pixels = render_pixels[idx]

  elif kwargs.get("shuffel", "random") == "distance_populating":
    # STEP0: Shuffle corrospondences
    idx = torch.randperm(P_ren.shape[0])
    P_ren = P_ren[idx]
    real_pixels = real_pixels[idx]
    render_pixels = render_pixels[idx]

    # STEP1: Bin values into grids
    u_bins = np.digitize(
      render_pixels[:, 0].numpy(),
      bins=np.arange(render_pixels[:, 0].min(), render_pixels[:, 0].max(), 5),
    )
    v_bins = np.digitize(
      render_pixels[:, 1].numpy(),
      bins=np.arange(render_pixels[:, 1].min(), render_pixels[:, 1].max(), 5),
    )

    indis_ori = np.arange(0, u_bins.shape[0])
    selected_points = []

    # STEP2: Iterate over every 2-th u-bin
    for u_bin in range(0, u_bins.max(), 2):
      # Create pixel mask for the bin.
      m = v_bins == u_bin
      s2_tmp = u_bins[m]
      indis_tmp = indis_ori[m]

      # STEP3: find unique indices in the v-bins with the u-bin mask applied
      a, indi = np.unique(s2_tmp, return_index=True)
      selection = indis_tmp[indi[::2]]
      # STEP4: append the corresponding indices of the orginale point cloud
      selected_points += selection.tolist()

    # STEP5: Fall back to random selection if necessary
    if len(selected_points) > kwargs.get("min_corrospondences", 30):
      P_ren = P_ren[selected_points]
      real_pixels = real_pixels[selected_points]
      render_pixels = render_pixels[selected_points]
    else:
      print(f"Sampling failed found {len( selected_points)} corrospondences")
      pts_trafo = min(P_ren.shape[0], kwargs.get("max_corrospondences", 50000))
      P_ren = P_ren[0:pts_trafo]
      real_pixels = real_pixels[0:pts_trafo]
      render_pixels = render_pixels[0:pts_trafo]
  else:
    raise ValueError(
      "Shuffle in flow_to_trafo not found", kwargs.get("shuffel", "random")
    )

  # Move the rendered points to the origin
  P_ren_in_origin = (
    get_H(P_ren).type(typ) @ torch.inverse(h_render.type(torch.float32)).type(typ).T
  )[:, :3]

  # PNP estimation
  objectPoints = P_ren_in_origin.cpu().type(torch.float32).numpy()
  imagePoints = real_pixels[:, :2].cpu().type(torch.float32).numpy()
  dist = np.array([[0.0, 0.0, 0.0, 0.0]])

  if objectPoints.shape[0] < 8:
    print(f"Failed due to missing corsspondences ({ objectPoints.shape[0]})")
    return (
      False,
      torch.eye(4, dtype=u_map.dtype, device=u_map.device),
      np.inf,
      0,
      Violation.MINIMAL_NR_VALID_CONSTRAINT,
    )
  # set current guess as the inital estimate

  rvec = R.from_matrix(h_real_est[:3, :3].cpu().numpy()).as_rotvec().astype(np.float32)
  tvec = h_real_est[:3, 3].cpu().numpy().astype(np.float32)
  # calculate PnP between the pixels coordinates in the real image and the corrosponding points in the origin frame

  if kwargs.get("method", "solvePnPRansac") == "solvePnPRansac":
    import time

    sta = time.time()
    for i in range(0, 100):
      retval, r_vec2, t_vec2, inliers = cv2.solvePnPRansac(
        objectPoints,
        imagePoints,
        cameraMatrix=K_real.cpu().type(torch.float32).numpy(),
        distCoeffs=dist,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        iterationsCount=kwargs.get("iterationsCount", 100),
        reprojectionError=kwargs.get("reprojectionError", 5),
        flags=kwargs.get("flags", 5),
      )
    sto = time.time()
    print("EPE", sto - sta)

  elif kwargs.get("method", "solvePnPRefineLM") == "solvePnPRefineLM":
    objP = copy.deepcopy(objectPoints)
    imgP = copy.deepcopy(imagePoints)
    K_rea = K_real.cpu().type(torch.float32).numpy()
    rvec_ = copy.deepcopy(rvec)[:, None]
    tvec_ = copy.deepcopy(tvec)[:, None]
    import time

    sta = time.time()
    lis = []
    for i in range(0, 100):
      r_vec2, t_vec2 = cv2.solvePnPRefineLM(
        objP,
        imgP,
        K_rea,
        dist,
        rvec_,
        tvec_,
      )
    sto = time.time()
    print("LM", sto - sta)

  elif kwargs.get("method", "solvePnPRefineLM") == "solveBoth":
    retval, r_vec2, t_vec2, inliers = cv2.solvePnPRansac(
      objectPoints,
      imagePoints,
      cameraMatrix=K_real.cpu().type(torch.float32).numpy(),
      distCoeffs=dist,
      rvec=rvec,
      tvec=tvec,
      useExtrinsicGuess=True,
      iterationsCount=kwargs.get("iterationsCount", 100),
      reprojectionError=kwargs.get("reprojectionError", 5),
      flags=kwargs.get("flags", 5),
    )
    r_vec2, t_vec2 = cv2.solvePnPRefineLM(
      copy.deepcopy(objectPoints),
      copy.deepcopy(imagePoints),
      K_real.cpu().type(torch.float32).numpy(),
      dist,
      copy.deepcopy(r_vec2),
      copy.deepcopy(t_vec2),
    )
  else:
    raise ValueError("NotDefined")

  h = rvec_tvec_to_H(r_vec2[:, 0], t_vec2)

  # calculate reprojection error
  imagePointsEst, jac = cv2.projectPoints(
    objectPoints[None], r_vec2, t_vec2, K_real.cpu().type(torch.float32).numpy(), dist
  )
  repro_error = np.linalg.norm(
    imagePointsEst[:, 0, :] - imagePoints, ord=2, axis=1
  ).mean()
  ratio = (
    np.linalg.norm(imagePointsEst[:, 0, :] - imagePoints, ord=2, axis=1)
    < kwargs.get("reprojectionError", 5)
  ).sum() / objectPoints.shape[0]

  return (
    True,
    torch.tensor(h, device=u_map.device).type(u_map.dtype),
    repro_error,
    ratio,
    Violation.SUCCESS,
  )
