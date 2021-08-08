import random
import copy
import os
import pickle
from pathlib import Path
from PIL import Image

# Frameworks
import numpy as np
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R
import scipy.io as scio
import torchvision.transforms as transforms
import torch

# For flow calculation
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import scipy.ndimage as nd

# From costume modules
from src.common.rotations import *

from src.common.ycb.ycb_helper import get_bb_from_depth, get_bb_real_target
from src.common.ycb.ycb_helper import Augmentation
from src.common.ycb.ycb_helper import ViewpointManager
from src.common.ycb.ycb_helper import backproject_points

from torch import from_numpy as fn

__all__ = "YCB"


class YCB(torch.utils.data.Dataset):
  def __init__(self, root, mode, image_size, cfg_d):
    self.estimate_pose = False
    self.fake_flow = False
    self._h = image_size[0]
    self._w = image_size[1]
    self._cfg_d = cfg_d
    self._load(mode, root)
    self._pcd_cad_list = self._get_pcd_cad_models(root)

    self._aug = Augmentation(
      add_depth=cfg_d.get("add_depth", False),
      output_size=(self._h, self._w),
      input_size=(self._h, self._w),
    )

    self._trancolor_background = transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)

    if mode == "test" or mode == "val":
      self._trancolor = transforms.ColorJitter(0.01, 0.01, 0.01, 0.005)
    else:
      ccj = cfg_d["aug_params"].get(
        "color_jitter",
        {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.05},
      )
      self._trancolor = transforms.ColorJitter(**ccj)

    self._vm = ViewpointManager(
      store=os.path.join(root, "viewpoints_renderings"),
      name_to_idx=self._names_idx,
      nr_of_images_per_object=cfg_d.get("nr_of_discrete_viewpoints", 10000),
      device="cpu",
      load_images=False,
    )

    self.K = {
      "1": np.array([[1077.836, 0, 323.7872], [0, 1078.189, 279.6921], [0, 0, 1]]),
      "0": np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]]),
    }
    self.K_ren = self.K["0"]

    self._load_flow(root)
    self.err = False
    self._num_pt_cad_model = 2600
    self.segmentation_only = False
    self.if_err_ret_none = False
    self.valid_flow_minimum = 100

    if self._cfg_d.get("init_mode", "pose_cnn") == "pose_cnn":

      with open("cfg/datasets/ycb/data_posecnn.pickle", "rb") as handle:
        self._posecnn_data = pickle.load(handle)

    self._base_path_list_background = copy.deepcopy(self._base_path_list)

    # self._obj_idx_list values 1-21 starting at 1
    if not self._cfg_d.get("filter", None) is None:
      keep = []
      for i in range(0, len(self._base_path_list)):
        keep.append(self._obj_idx_list[i] in self._cfg_d["filter"])

      keep = np.array(keep)
      print(
        "Filter dataset: ",
        self._cfg_d["filter"],
        " from ",
        len(self._base_path_list),
        " to ",
        keep.sum(),
      )
      self._base_path_list = np.array(self._base_path_list)[keep].tolist()
      self._obj_idx_list = np.array(self._obj_idx_list)[keep].tolist()
      self._camera_idx_list = np.array(self._camera_idx_list)[keep].tolist()

      self._length = len(self._base_path_list)

  def _load(self, mode, root):
    with open(f"cfg/datasets/ycb/{mode}.pkl", "rb") as handle:
      mappings = pickle.load(handle)
      self._names_idx = mappings["names_idx"]
      self._idx_names = mappings["idx_names"]
      self._base_path_list = mappings["base_path_list"]
      self._base_path_list = [os.path.join(root, p) for p in self._base_path_list]
      self._obj_idx_list = mappings["obj_idx_list"]
      self._camera_idx_list = mappings["camera_idx_list"]
      self._length = len(self._base_path_list)

    # CHANGING RATIO OF SYNTHETIC DATA BY DUPLICATING SYNTHETIC LABELS
    ratio = self._cfg_d.get("ratio", None)
    if not (ratio is None):
      # sum_val = 0
      # for v in ratio.values():
      #   sum_val += v
      # for k in ratio.keys():
      #   ratio[k] /= sum_val

      idxs = {"data": [], "data_syn": [], "data_syn_new": []}
      for j, p in enumerate(self._base_path_list):
        for k in idxs.keys():
          if p.find(f"/{k}/") != -1:
            idxs[k].append(j)

      multiply = {}
      repeats = []  # contains all the indices that should be repeated.
      for id in ["data_syn", "data_syn_new"]:
        multiply[id] = len(idxs["data"]) / len(idxs[id]) * ratio[id]

        while multiply[id] > 1:
          # Add all
          repeats = repeats + idxs[id]

          multiply[id] -= 1

        elements_to_sample = int(multiply[id] * len(idxs[id]))
        if multiply[id] > 0 and elements_to_sample > 0:
          repeats = (
            repeats
            + (
              np.array(idxs[id])[
                np.random.permutation(len(idxs[id]))[:elements_to_sample]
              ]
            ).tolist()
          )
      repeats = repeats + idxs["data"]

      self._base_path_list = np.array(self._base_path_list)[repeats].tolist()
      self._obj_idx_list = np.array(self._obj_idx_list)[repeats].tolist()
      self._camera_idx_list = np.array(self._camera_idx_list)[repeats].tolist()

      self._length = len(self._base_path_list)

      idxs = {"data": [], "data_syn": [], "data_syn_new": []}
      for j, p in enumerate(self._base_path_list):
        for k in idxs.keys():
          if p.find(f"/{k}/") != -1:
            idxs[k].append(j)
      print(
        "OUPUT RATIOS",
        len(idxs["data"]),
        len(idxs["data_syn"]),
        len(idxs["data_syn_new"]),
      )

  def __getitem__(self, index):
    return self.getElement(index, h_real_est=None)

  def _get_background_image(self, obj_idx):
    while 1:
      index = random.randint(0, len(self._base_path_list_background) - 1)
      p = self._base_path_list_background[index]
      if p.find("data_syn") != -1:
        continue
      meta = scio.loadmat(p + "-meta.mat")
      obj = meta["cls_indexes"].flatten().astype(np.int32)
      if not obj_idx in obj:
        break

    img = Image.open(p + "-color.png").convert("RGB")
    w, h = img.size
    w_g, h_g = 640, 480
    if w / h < w_g / h_g:
      h = int(w * h_g / w_g)
    else:
      w = int(h * w_g / h_g)
    crop = transforms.CenterCrop((h, w))
    img = crop(img)
    img = img.resize((w_g, h_g))
    return np.array(self._trancolor_background(img))

  def deterministic_random_shuffel(self):
    torch.manual_seed(42)
    idx = torch.randperm(len(self._base_path_list)).numpy()
    self._base_path_list = np.array(self._base_path_list)[idx].tolist()
    self._obj_idx_list = np.array(self._obj_idx_list)[idx].tolist()
    self._camera_idx_list = np.array(self._camera_idx_list)[idx].tolist()
    print("Shuffeld the dataset")

  def getElement(self, index, h_real_est=None):
    """
    desig : sequence/idx
    two problems we face. What is if an object is not visible at all -> meta['obj'] = None
    obj_idx is elemnt 1-21 !!!
    """
    p = self._base_path_list[index]
    obj_idx = self._obj_idx_list[index]
    K = self.K[str(self._camera_idx_list[index])]
    synthetic = p.find("syn") != -1

    img = Image.open(p + "-color.png")
    depth = np.array(Image.open(p + "-depth.png"))
    label = np.array(Image.open(p + "-label.png"))
    meta = scio.loadmat(p + "-meta.mat")
    obj = meta["cls_indexes"].flatten().astype(np.int32)

    if p.find("data_syn_new") != -1:
      obj_idx_in_list = 0
    else:
      obj_idx_in_list = int(np.argwhere(obj == obj_idx))

    h_gt = np.eye(4)
    h_gt[:3, :4] = meta["poses"][:, :, obj_idx_in_list]
    h_gt = h_gt.astype(np.float32)

    if synthetic:
      img_arr = np.array(img)[:, :, :3]
      background_img = self._get_background_image(obj_idx)
      mask = label == 0
      img_arr[mask] = background_img[mask]
    else:
      img_arr = np.array(img)[:, :, :3]

    if self.estimate_pose:
      img_ori = torch.from_numpy(copy.deepcopy(img_arr))

    dellist = [j for j in range(0, len(self._pcd_cad_list[obj_idx - 1]))]
    dellist = random.sample(
      dellist, len(self._pcd_cad_list[obj_idx - 1]) - self._num_pt_cad_model
    )
    model_points = np.delete(self._pcd_cad_list[obj_idx - 1], dellist, axis=0).astype(
      np.float32
    )

    cam_flag = self._camera_idx_list[index]
    idx = torch.LongTensor([int(obj_idx) - 1])

    if h_real_est is None:
      m = self._cfg_d.get("init_mode", "pose_cnn")
      if m == "pose_cnn":
        h_real_est = self._get_init_pose_posecnn(obj_idx, p)
      elif m == "tracking":
        h_real_est = get_init_pose_track(
          obj_idx, p, h_gt, model_points, K, size=(self._h, self._w)
        )
      elif m == "tracking_gt":
        h_real_est = get_init_pose_track_gt(
          obj_idx, p, h_gt, model_points, K, size=(self._h, self._w)
        )
      elif m == "noise_adaptive":
        h_real_est = get_adaptive_noise(
          model_points,
          h_gt,
          K,
          obj_idx=obj_idx,
          factor=5,
          rot_deg=self._cfg_d["output_cfg"].get("noise_rotation", 30),
        )

      if h_real_est is None and self.if_err_ret_none:
        print("PoseCNN failed")

        if self.if_err_ret_none:
          return (None, idx)
        else:
          new_idx = random.randint(0, len(self) - 1)
          return self[new_idx]

      if m == "noise" or h_real_est is None:
        nt = self._cfg_d["output_cfg"].get("noise_translation", 0.02)
        nr = self._cfg_d["output_cfg"].get("noise_rotation", 30)
        h_real_est = add_noise(h_gt, nt, nr)

    try:
      res_get_render = self.get_rendered_data(
        img_arr, depth, label, model_points, int(obj_idx), K, cam_flag, h_gt, h_real_est
      )
    except:
      res_get_render = False

    if res_get_render is False:
      if self.err:
        print("Violation in get render data")

      if self.if_err_ret_none:
        return (None, idx)
      else:
        new_idx = random.randint(0, len(self) - 1)
        return self[new_idx]

    real = res_get_render[0]
    render = res_get_render[1]
    # AUGMENTATION
    real = self._trancolor(real.permute(2, 0, 1) / 255)

    if self._cfg_d["aug_params"].get("color_jitter_render", True):
      render = self._trancolor(render.permute(2, 0, 1) / 255)
    else:
      render = render.permute(2, 0, 1) / 255

    if self.segmentation_only:
      return (real, render, res_get_render[2].type(torch.long), torch.tensor(synthetic))

    real *= 255.0
    render *= 255.0
    flow = torch.cat(
      [res_get_render[5][:, :, None], res_get_render[6][:, :, None]], dim=2
    )

    # TEMPLATE INTERFACE
    flow = flow.numpy().astype(np.float32)  # H,W,
    valid = res_get_render[7].numpy().astype(np.float32)
    if valid is not None:
      valid = torch.from_numpy(valid)
    else:
      valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
    flow = fn(flow).permute(2, 0, 1)

    if self.estimate_pose:
      h_render = res_get_render[9]  # 4,4
      h_init = res_get_render[10]  # 4,4
      bb = res_get_render[8]
      K_real = res_get_render[-3]  # 3,3
      K_ren = self.K_ren  # 3,3
      render_d = res_get_render[3]  # H,W
      model_points = model_points  # NR, 3
      img_render_ori = res_get_render[-2]

      return (
        real,
        render,
        flow,
        valid.float(),
        torch.tensor(synthetic),
        torch.from_numpy(h_gt),
        h_render,
        h_init,
        bb,
        idx,
        torch.from_numpy(K_ren),
        K_real,
        render_d,
        torch.from_numpy(model_points),
        img_ori,
        p,
        img_render_ori,
      )
    else:
      bb = res_get_render[8]
      return (real, render, flow, valid.float(), torch.tensor(synthetic), idx, bb)

  def get_rendered_data(
    self,
    img,
    depth_real,
    label,
    model_points,
    obj_idx,
    K_real,
    cam_flag,
    h_gt,
    h_real_est=None,
  ):
    """Get Rendered Data
    Args:
      img ([np.array numpy.uint8]): H,W,3
      depth_real ([np.array numpy.int32]): H,W
      label ([np.array numpy.uint8]): H,W
      model_points ([np.array numpy.float32]): 2300,3
      obj_idx: (Int)
      K_real ([np.array numpy.float32]): 3,3
      cam_flag (Bool)
      h_gt ([np.array numpy.float32]): 4,4
      h_real_est ([np.array numpy.float32]): 4,4
    Returns:
      real_img ([torch.tensor torch.float32]): H,W,3
      render_img ([torch.tensor torch.float32]): H,W,3
      real_d ([torch.tensor torch.float32]): H,W
      render_d ([torch.tensor torch.float32]): H,W
      gt_label_cropped ([torch.tensor torch.long]): H,W
      u_cropped_scaled ([torch.tensor torch.float32]): H,W
      v_cropped_scaled([torch.tensor torch.float32]): H,W
      valid_flow_mask_cropped([torch.tensor torch.bool]): H,W
      bb ([tuple]) containing torch.tensor( real_tl, dtype=torch.int32) , torch.tensor( real_br, dtype=torch.int32) , torch.tensor( ren_tl, dtype=torch.int32) , torch.tensor( ren_br, dtype=torch.int32 )
      h_render ([torch.tensor torch.float32]): 4,4
      h_init ([torch.tensor torch.float32]): 4,4
    """
    h = self._h
    w = self._w

    output_h = self._h
    output_w = self._w

    h_init = h_real_est

    # transform points
    rot = R.from_euler("z", 180, degrees=True).as_matrix()
    pred_points = (model_points @ h_init[:3, :3].T) + h_init[:3, 3]

    init_rot_wxyz = re_quat(
      torch.from_numpy(R.from_matrix(h_init[:3, :3]).as_quat()), "xyzw"
    )
    idx = torch.LongTensor([int(obj_idx) - 1])

    img_ren, depth_ren, h_render = self._vm.get_closest_image_batch(
      i=idx[None], rot=init_rot_wxyz, conv="wxyz"
    )

    # rendered data BOUNDING BOX Computation
    bb_lsd = get_bb_from_depth(depth_ren)
    b_ren = bb_lsd[0]
    tl, br = b_ren.limit_bb()
    if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b_ren.violation():
      if self.err:
        print("Violate BB in get render data for rendered bb")
      return False
    center_ren = backproject_points(h_render[0, :3, 3].view(1, 3), K=self.K_ren)
    center_ren = center_ren.squeeze()
    b_ren.move(-center_ren[1], -center_ren[0])
    b_ren.expand(self._cfg_d.get("expand_factor", 1.3))
    b_ren.expand_to_correct_ratio(output_w, output_h)
    b_ren.move(center_ren[1], center_ren[0])
    ren_h = b_ren.height()
    ren_w = b_ren.width()
    ren_tl = b_ren.tl
    if (
      ren_h < 20
      or ren_w < 20
      or img_ren.shape[1] < 20
      or img_ren.shape[2] < 20
      or depth_ren.shape[1] < 20
      or depth_ren.shape[2] < 20
    ):
      print("img_ren", img_ren.shape, depth_ren.shape)
      return False

    render_img = b_ren.crop(
      img_ren[0], scale=True, mode="bilinear", output_h=output_h, output_w=output_w
    )  # Input H,W,C
    render_d = b_ren.crop(
      depth_ren[0][:, :, None],
      scale=True,
      mode="nearest",
      output_h=output_h,
      output_w=output_w,
    )  # Input H,W,C

    # real data BOUNDING BOX Computation
    bb_lsd = get_bb_real_target(torch.from_numpy(pred_points[None, :, :]), K_real[None])
    b_real = bb_lsd[0]
    tl, br = b_real.limit_bb()
    if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b_real.violation():
      if self.err:
        print("Violate BB in get render data for real bb")
      return False
    center_real = backproject_points(torch.from_numpy(h_init[:3, 3][None]), K=K_real)
    center_real = center_real.squeeze()

    b_real.move(-center_real[0], -center_real[1])
    b_real.expand(self._cfg_d.get("expand_factor", 1.3))
    b_real.expand_to_correct_ratio(output_w, output_h)
    b_real.move(center_real[0], center_real[1])
    real_h = b_real.height()
    real_w = b_real.width()
    real_tl = b_real.tl

    if (
      real_h < 20
      or real_w < 20
      or img.shape[0] < 20
      or img.shape[1] < 20
      or depth_real.shape[0] < 20
      or depth_real.shape[1] < 20
      or label.shape[0] < 20
      or label.shape[1] < 20
    ):
      print("idl", img.shape, depth_real.shape, label.shape)
      return False

    real_img = b_real.crop(
      torch.from_numpy(img).type(torch.float32),
      scale=True,
      mode="bilinear",
      output_h=output_h,
      output_w=output_w,
    )

    real_d = b_real.crop(
      torch.from_numpy(depth_real[:, :, None]).type(torch.float32),
      scale=True,
      mode="nearest",
      output_h=output_h,
      output_w=output_w,
    )
    gt_label_cropped = b_real.crop(
      torch.from_numpy(label[:, :, None]).type(torch.float32),
      scale=True,
      mode="nearest",
      output_h=output_h,
      output_w=output_w,
    ).type(torch.int32)
    # LGTM

    if self.fake_flow:
      real_tl = np.zeros((2))
      real_tl[0] = int(b_real.tl[0])
      real_tl[1] = int(b_real.tl[1])
      real_br = np.zeros((2))
      real_br[0] = int(b_real.br[0])
      real_br[1] = int(b_real.br[1])
      ren_tl = np.zeros((2))
      ren_tl[0] = int(b_ren.tl[0])
      ren_tl[1] = int(b_ren.tl[1])
      ren_br = np.zeros((2))
      ren_br[0] = int(b_ren.br[0])
      ren_br[1] = int(b_ren.br[1])

      bbs = (
        torch.tensor(real_tl, dtype=torch.int32),
        torch.tensor(real_br, dtype=torch.int32),
        torch.tensor(ren_tl, dtype=torch.int32),
        torch.tensor(ren_br, dtype=torch.int32),
      )

      ls = [
        real_img,
        render_img,
        real_d[:, :, 0],
        render_d[:, :, 0],
        gt_label_cropped.type(torch.long)[:, :, 0],
        torch.zeros_like(real_d).type(torch.float32)[:, :, 0],
        torch.zeros_like(real_d).type(torch.float32)[:, :, 0],
        torch.zeros_like(real_d).type(torch.long)[:, :, 0],
        bbs,
        h_render[0].type(torch.float32),
        torch.from_numpy(h_init).type(torch.float32),
        torch.from_numpy(h_gt).type(torch.float32),
        torch.from_numpy(K_real.astype(np.float32)),
        img_ren[0],
        depth_ren[0],
      ]
      return ls

    flow = self._get_flow_fast(
      h_render[0].numpy(),
      h_gt,
      obj_idx,
      label,
      cam_flag,
      b_real,
      b_ren,
      K_real,
      depth_ren[0],
      output_h,
      output_w,
    )
    if flow is False:
      return False

    valid_flow_mask_cropped = (
      b_real.crop(
        torch.from_numpy(flow[2][:, :, None]).type(torch.float32),
        scale=True,
        mode="nearest",
        output_h=output_h,
        output_w=output_w,
      )
      .type(torch.bool)
      .numpy()
    )
    if self.segmentation_only:
      return real_img, render_img, torch.from_numpy(valid_flow_mask_cropped[:, :, 0])

    if flow[2].sum() < self.valid_flow_minimum:
      return False

    u_cropped = b_real.crop(
      torch.from_numpy(flow[0][:, :, None]).type(torch.float32),
      scale=True,
      mode="bilinear",
      output_h=output_h,
      output_w=output_w,
    ).numpy()
    v_cropped = b_real.crop(
      torch.from_numpy(flow[1][:, :, None]).type(torch.float32),
      scale=True,
      mode="bilinear",
      output_h=output_h,
      output_w=output_w,
    ).numpy()

    # scale the u and v so this is not in the uncropped space !
    _grid_x, _grid_y = np.mgrid[0:output_h, 0:output_w].astype(np.float32)

    nr1 = np.full((output_h, output_w), float(output_w / real_w), dtype=np.float32)
    nr2 = np.full((output_h, output_w), float(real_tl[1]), dtype=np.float32)
    nr3 = np.full((output_h, output_w), float(ren_tl[1]), dtype=np.float32)
    nr4 = np.full((output_h, output_w), float(output_w / ren_w), dtype=np.float32)
    v_cropped_scaled = _grid_y - (
      (np.multiply(((np.divide(_grid_y, nr1) + nr2) + (v_cropped[:, :, 0])) - nr3, nr4))
    )

    nr1 = np.full((output_h, output_w), float(output_h / real_h), dtype=np.float32)
    nr2 = np.full((output_h, output_w), float(real_tl[0]), dtype=np.float32)
    nr3 = np.full((output_h, output_w), float(ren_tl[0]), dtype=np.float32)
    nr4 = np.full((output_h, output_w), float(output_h / ren_h), dtype=np.float32)
    u_cropped_scaled = _grid_x - (
      np.round((((_grid_x / nr1) + nr2) + np.round(u_cropped[:, :, 0])) - nr3) * (nr4)
    )

    ls = [
      real_img,
      render_img,
      real_d[:, :, 0],
      render_d[:, :, 0],
      gt_label_cropped.type(torch.long)[:, :, 0],
      torch.from_numpy(u_cropped_scaled[:, :]).type(torch.float32),
      torch.from_numpy(v_cropped_scaled[:, :]).type(torch.float32),
      torch.from_numpy(valid_flow_mask_cropped[:, :, 0]),
      flow[-4:],
      h_render[0].type(torch.float32),
      torch.from_numpy(h_init).type(torch.float32),
      torch.from_numpy(h_gt).type(torch.float32),
      torch.from_numpy(K_real.astype(np.float32)),
      img_ren[0],
      depth_ren[0],
    ]

    return ls

  def _get_flow_fast(
    self,
    h_render,
    h_real,
    idx,
    label_img,
    cam,
    b_real,
    b_ren,
    K_real,
    render_d,
    output_h,
    output_w,
  ):
    m_real = copy.deepcopy(self._mesh[idx])
    m_real = transform_mesh(m_real, h_real)

    rmi_real = RayMeshIntersector(m_real)
    tl, br = b_real.limit_bb()
    rays_origin_real = self._rays_origin_real[cam][
      int(tl[0]) : int(br[0]), int(tl[1]) : int(br[1])
    ]
    rays_dir_real = self._rays_dir[cam][
      int(tl[0]) : int(br[0]), int(tl[1]) : int(br[1])
    ]

    real_locations, real_index_ray, real_res_mesh_id = rmi_real.intersects_location(
      ray_origins=np.reshape(rays_origin_real, (-1, 3)),
      ray_directions=np.reshape(rays_dir_real, (-1, 3)),
      multiple_hits=False,
    )

    h_real_inv = np.eye(4)
    h_real_inv[:3, :3] = h_real[:3, :3].T
    h_real_inv[:3, 3] = -h_real_inv[:3, :3] @ h_real[:3, 3]
    h_trafo = h_render @ h_real_inv

    ren_locations = (copy.deepcopy(real_locations) @ h_trafo[:3, :3].T) + h_trafo[:3, 3]
    uv_ren = backproject_points_np(ren_locations, K=self.K_ren)
    index_the_depth_map = np.round(uv_ren)

    val = (index_the_depth_map[:, 0] < 480) * (index_the_depth_map[:, 0] >= 0)
    val2 = (index_the_depth_map[:, 1] < 640) * (index_the_depth_map[:, 1] >= 0)

    v = val * val2
    if v.sum() < self.valid_flow_minimum:
      return False

    new_tensor = render_d[index_the_depth_map[v, 0], index_the_depth_map[v, 1]] / 10000
    distance_depth_map_to_model = torch.abs(
      new_tensor[:] - torch.from_numpy(ren_locations[v, 2])
    )
    not_val = distance_depth_map_to_model > 0.005

    valid_points_for_flow = v

    if not_val.sum() != 0:
      valid_points_for_flow[v == True][not_val] = False

    uv_real = backproject_points_np(real_locations, K=K_real)

    valid_flow_index = uv_real[valid_points_for_flow].astype(np.uint32)
    valid_flow = np.zeros((label_img.shape[0], label_img.shape[1]))
    valid_flow[valid_flow_index[:, 0], valid_flow_index[:, 1]] = 1

    dis = uv_ren - uv_real
    uv_real = np.uint32(uv_real)
    idx_ = np.uint32(uv_real[:, 0] * (self._w) + uv_real[:, 1])

    disparity_pixels = np.zeros((self._h, self._w, 2)) - 999
    disparity_pixels = np.reshape(disparity_pixels, (-1, 2))
    disparity_pixels[idx_] = dis
    disparity_pixels = np.reshape(disparity_pixels, (self._h, self._w, 2))

    u_map = disparity_pixels[:, :, 0]
    v_map = disparity_pixels[:, :, 1]
    u_map = fill(u_map, u_map == -999)
    v_map = fill(v_map, v_map == -999)

    real_tl = np.zeros((2))
    real_tl[0] = int(b_real.tl[0])
    real_tl[1] = int(b_real.tl[1])
    real_br = np.zeros((2))
    real_br[0] = int(b_real.br[0])
    real_br[1] = int(b_real.br[1])
    ren_tl = np.zeros((2))
    ren_tl[0] = int(b_ren.tl[0])
    ren_tl[1] = int(b_ren.tl[1])
    ren_br = np.zeros((2))
    ren_br[0] = int(b_ren.br[0])
    ren_br[1] = int(b_ren.br[1])

    f_3 = valid_flow
    f_3 *= label_img == idx
    return (
      u_map,
      v_map,
      f_3,
      torch.tensor(real_tl, dtype=torch.int32),
      torch.tensor(real_br, dtype=torch.int32),
      torch.tensor(ren_tl, dtype=torch.int32),
      torch.tensor(ren_br, dtype=torch.int32),
    )

  def __len__(self):
    return self._length

  def _load_flow(self, root):
    self._load_rays_dir()
    self._load_meshes(root)

    self._max_matches = self._cfg_d.get("flow_cfg", {}).get("max_matches", 1500)
    self._max_iterations = self._cfg_d.get("flow_cfg", {}).get("max_iterations", 10000)
    self._grid_x, self._grid_y = np.mgrid[0 : self._h, 0 : self._w]

  def _load_rays_dir(self):
    self._rays_origin_real = []
    self._rays_origin_render = []
    self._rays_dir = []

    for K in [self.K["0"], self.K["1"]]:
      u_cor = np.arange(0, self._h, 1)
      v_cor = np.arange(0, self._w, 1)
      K_inv = np.linalg.inv(K)
      rays_dir = np.zeros((self._w, self._h, 3))
      nr = 0
      rays_origin_render = np.zeros((self._w, self._h, 3))
      rays_origin_real = np.zeros((self._w, self._h, 3))
      for u in v_cor:
        for v in u_cor:
          n = K_inv @ np.array([u, v, 1])
          # n = np.array([n[1],n[0],n[2]])
          rays_dir[u, v, :] = n * 0.6 - n * 0.25
          rays_origin_render[u, v, :] = n * 0.1
          rays_origin_real[u, v, :] = n * 0.25
          nr += 1
      rays_origin_render
      self._rays_origin_real.append(np.swapaxes(rays_origin_real, 0, 1))
      self._rays_origin_render.append(np.swapaxes(rays_origin_render, 0, 1))
      self._rays_dir.append(np.swapaxes(rays_dir, 0, 1))

  def _load_meshes(self, root):
    p = os.path.join(root, "models")
    cad_models = [str(p) for p in Path(p).rglob("*scaled.obj")]  # textured
    self._mesh = {}
    for pa in cad_models:
      idx = self._names_idx[pa.split("/")[-2]]
      self._mesh[idx] = trimesh.load(pa)

  def _get_pcd_cad_models(self, root):
    cad_paths = []
    for n in self._names_idx.keys():
      cad_paths.append(root + "/models/" + n)

    cad_list = []
    for path, names in zip(cad_paths, list(self._names_idx.keys())):
      input_file = open("{0}/points.xyz".format(path))

      cld = []
      while 1:
        input_line = input_file.readline()
        if not input_line:
          break
        input_line = input_line[:-1].split(" ")
        cld.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
      cad_list.append(np.array(cld))
      input_file.close()

    return cad_list

  def _get_init_pose_posecnn(self, obj_idx, p):
    seq_id = int(p.split("/")[-2])
    frame_id = int(p.split("/")[-1])
    look_back_distance = 40
    seq_ids = np.array([d["seq_id"] for d in self._posecnn_data])
    frame_ids = np.array([d["Frame_id"] for d in self._posecnn_data])
    cls_ids = np.array([d["cls_index"] for d in self._posecnn_data])

    for i in range(look_back_distance):
      a = (seq_ids == seq_id) * (frame_ids == frame_id - i) * (cls_ids == obj_idx)

      if a.sum() != 1:
        if i == (look_back_distance - 1):
          print(f"INITAL POSE WAS NOT FOUND even though looked back {i}")
          return None
        else:
          continue
      else:
        break

    idx = np.where(a)[0][0]
    h = np.eye(4)
    h[:3] = self._posecnn_data[idx]["H"]
    return h


def get_init_pose_track(obj_idx, p, h_gt, model_points, K, size):
  key = p + "/" + str(obj_idx)
  for i in range(1, 50):
    k_tmp = key.split("/")
    nr = int(k_tmp[-2]) - i
    k_tmp[-2] = f"{nr:06d}"
    k = "/".join(k_tmp)
    tmp = os.path.join("/home/jonfrey/tmp", k[k.find("ycb") :] + ".npy")
    if os.path.isfile(tmp):
      print("FOUND", p, obj_idx)
      print("WILL RETURN ", tmp)
      res = np.load(tmp)

      if np.linalg.norm(res[:3, 3] - h_gt[:3, 3]) < 0.03:
        return res
      else:
        break

  print("NOT FOUND ", p, obj_idx)
  return get_init_pose_posecnn(obj_idx, p, h_gt, model_points, K, size)


def get_init_pose_track_gt(obj_idx, p, h_gt, model_points, K, size):
  key = p + "/" + str(obj_idx)
  try:
    for i in range(1, 50):
      k_tmp = key.split("/")
      nr = int(k_tmp[-2]) - i
      k_tmp[-2] = f"{nr:06d}"
      k = "/".join(k_tmp)
      tmp = os.path.join("/home/jonfrey/tmp", k[k.find("ycb") :] + ".npy")
      if os.path.isfile(tmp):
        res = np.load(tmp)

        if np.linalg.norm(res[:3, 3] - h_gt[:3, 3]) < 0.1:
          return res
        else:
          break
  except:
    pass
  print("NOT FOUND ", p, obj_idx)
  return h_gt


def get_init_pose_posecnn(obj_idx, p, h_gt, model_points, K, size):

  h_real_est = None
  base = "/home/jonfrey/PoseCNN-PyTorch/output/ycb_video/ycb_video_keyframe/vgg16_ycb_video_epoch_16.checkpoint.pth/"
  m = os.path.join(base, p.split("/")[-2] + "_" + p.split("/")[-1] + ".mat")
  result = scio.loadmat(m)
  pcnn_class_idxs = result["rois"][:, 1]
  possible_rois_idx = np.where(pcnn_class_idxs == obj_idx)[0].tolist()
  target = model_points @ h_gt[:3, :3].T + h_gt[:3, 3]
  bb = get_bb_real_target(torch.from_numpy(target[None, :, :]), K[None])[0]
  bb_gt = np.zeros(size, dtype=bool)
  bb_gt[
    int(bb.tl[0]) : int(bb.br[0]), int(bb.tl[1]) : int(bb.br[1])
  ] = True  # BINARY mask over BB

  if len(possible_rois_idx) > 0:
    # iterate over possible rois and find one with highest overlap
    overlaps = []
    for rois_idx in possible_rois_idx:
      bb_pcnn = np.zeros(size, dtype=bool)
      tl = (result["rois"][rois_idx, 3], result["rois"][rois_idx, 2])
      br = (result["rois"][rois_idx, 5], result["rois"][rois_idx, 4])
      bb_pcnn[
        int(tl[0]) : int(br[0]), int(tl[1]) : int(br[1])
      ] = True  # BINARY mask over BB
      overlaps.append((bb_gt * bb_pcnn).sum() / (bb_gt + bb_pcnn).sum())  # IoU

    max_rois = np.array(overlaps).argmax()
    if overlaps[max_rois] > 0.5:
      # successfull selected
      matched_roi = possible_rois_idx[max_rois]
      quat = result["poses"][matched_roi, :4]  # Nx7
      trans = result["poses"][matched_roi, -3:]  # Nx7
      h_real_est = np.eye(4)
      h_real_est[:3, 3] = trans
      h_real_est[:3, :3] = quat_to_rot(
        torch.from_numpy(quat)[None], conv="wxyz", device="cpu"
      ).numpy()
  return h_real_est


def transform_mesh(mesh, H):
  """directly operates on mesh and does not create a copy!"""
  t = np.ones((mesh.vertices.shape[0], 4))
  t[:, :3] = mesh.vertices
  H[:3, :3] = H[:3, :3]
  mesh.vertices = (t @ H.T)[:, :3]
  return mesh


def rel_h(h1, h2):
  return so3_relative_angle(
    torch.tensor(h1)[:3, :3][None], torch.tensor(h2)[:3, :3][None]
  )


def add_noise(h, nt=0.01, nr=30):
  h_noise = np.eye(4)
  while True:
    x = special_ortho_group.rvs(3).astype(np.float32)
    # _noise[:3,:3] = R.from_euler('zyx', np.random.uniform( -nr, nr, (1, 3) ) , degrees=True).as_matrix()[0]
    if abs(float(rel_h(h[:3, :3], x) / (2 * float(np.math.pi)) * 360)) < nr:
      break
  h_noise[:3, :3] = x
  h_noise[:3, 3] = np.random.normal(loc=h[:3, 3], scale=nt)

  return h_noise


def fill(data, invalid=None):
  """
  Replace the value of invalid 'data' cells (indicated by 'invalid')
  by the value of the nearest valid data cell
  Input:
    data:    numpy array of any dimension
    invalid: a binary array of same shape as 'data'. True cells set where data
         value should be replaced.
         If None (default), use: invalid  = np.isnan(data)
  Output:
    Return a filled array.
  """
  if invalid is None:
    invalid = np.isnan(data)
  ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
  return data[tuple(ind)]


def backproject_points_np(p, fx=None, fy=None, cx=None, cy=None, K=None):
  """
  p.shape = (nr_points,xyz)
  """
  if not K is None:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
  # true_divide
  u = ((p[:, 0] / p[:, 2]) * fx) + cx
  v = ((p[:, 1] / p[:, 2]) * fy) + cy
  return np.stack([v, u]).T


def expand(bb, h, w):
  bb.tl[0] = bb.tl[0] - h
  bb.tl[1] = bb.tl[1] - w
  bb.br[0] = bb.br[0] + h
  bb.br[1] = bb.br[1] + w


def calculate_bb_cone(K, bb, mean):
  points = np.stack([bb.tl.numpy(), bb.br.numpy()])
  points = np.concatenate([points, np.ones((2, 1))], axis=1)
  return (np.linalg.inv(K) @ points.T * mean).T


def get_adaptive_noise(model_points, h_gt, K, obj_idx=0, factor=5, rot_deg=30):
  target_points = model_points @ h_gt[:3, :3].T + h_gt[:3, 3]
  bb = get_bb_real_target(torch.from_numpy(target_points[None, :, :]), K[None])[0]
  h_, w_ = bb.height(), bb.width()
  bb_min = copy.deepcopy(bb)
  bb_max = copy.deepcopy(bb)
  expand(bb_min, h=-int(h_ / factor), w=-int(w_ / factor))
  expand(bb_max, h=int(h_ / factor), w=int(w_ / factor))

  mean_dis = np.mean(target_points[:, 2])
  mi = calculate_bb_cone(K, bb_min, mean_dis)
  ma = calculate_bb_cone(K, bb_max, mean_dis)
  a1 = mi - ma

  noise = (a1[0, 0], a1[0, 1], np.mean(a1[0, :2]) * 1.2)

  h_pred_est = np.eye(4)
  h_pred_est[:3, 3] = np.random.uniform(
    low=h_gt[:3, 3] - noise, high=h_gt[:3, 3] + noise, size=(3)
  )

  if obj_idx == 12:
    while True:
      x = (
        R.from_euler("xy", np.random.uniform(-180, 180, (2)), degrees=True)
        .as_matrix()
        .astype(np.float32)
        @ h_gt[:3, :3]
      )
      if abs(np.degrees(rel_h(h_gt[:3, :3], x))) < rot_deg:
        break

  while True:
    x = special_ortho_group.rvs(3).astype(np.float32)
    if abs(np.degrees(rel_h(h_gt[:3, :3], x))) < rot_deg:
      break
  h_pred_est[:3, :3] = x
  return h_pred_est
