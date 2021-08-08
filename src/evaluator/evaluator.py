import os
import sys

os.chdir(os.path.join(os.getenv("HOME"), "RPOSE"))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + "/src"))
sys.path.append(os.path.join(os.getcwd() + "/core"))
sys.path.append(os.path.join(os.getcwd() + "/segmentation"))

import coloredlogs

coloredlogs.install()
import shutil
import datetime
import argparse
from pathlib import Path
import os
import torch
from src_utils import file_path, load_yaml
import datasets
from lightning import Inferencer
from pose_estimation import full_pose_estimation, compute_auc
import numpy as np

from enum import Enum
from ycb.rotations import so3_relative_angle
from scipy.stats import special_ortho_group
import neptune.new as neptune
from pose_estimation import Violation
from ycb.ycb_helper import BoundingBox
import cv2
import time
from skimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import disk  # noqa


def expand_to_batch(batch, device):
  ret = []
  for b in batch:
    if torch.is_tensor(b):
      ret.append(b[None].cuda())
    elif type(b) is tuple:
      new = []
      for el in b:
        new.append(el[None].cuda())
      ret.append(tuple(new))
    else:
      ret.append(b)

  # return not mutable
  return tuple(ret)


class Mode(Enum):
  TRACKING = 1
  REFINEMENT = 2
  MUTIPLE_INIT_POSES = 3


def str_to_mode(s):
  if s == "MUTIPLE_INIT_POSES":
    return Mode.MUTIPLE_INIT_POSES
  elif s == "REFINEMENT":
    return Mode.REFINEMENT
  elif s == "TRACKING":
    return Mode.TRACKING
  else:
    raise Exception


def compute_tracklets(paths, objs):
  tracklets = []
  global_idx = np.arange(0, len(paths))

  for o in range(objs.min(), objs.max() + 1):
    p = paths[objs == o]
    local_idx = global_idx[objs == o]

    tracklets.append([])
    seq_last, idx_last = int(p[0].split("/")[-2]), int(p[0].split("/")[-1]) - 1

    for j, _p in enumerate(p):
      seq_cur, idx_cur = int(_p.split("/")[-2]), int(_p.split("/")[-1])
      if seq_cur == seq_last and idx_cur - 50 < idx_last and idx_cur > idx_last:
        tracklets[-1].append(local_idx[j])
      else:
        tracklets.append([local_idx[j]])
      seq_last, idx_last = seq_cur, idx_cur
  return tracklets


def rel_h(h1, h2):
  return so3_relative_angle(
    torch.tensor(h1)[:3, :3][None], torch.tensor(h2)[:3, :3][None]
  )


def add_noise(h, nt=0.01, nr=30):
  h_noise = np.eye(4)
  while True:
    x = special_ortho_group.rvs(3).astype(np.float32)
    if abs(float(rel_h(h[:3, :3], x) / (2 * float(np.math.pi)) * 360)) < nr:
      break
  h_noise[:3, :3] = x
  h_noise[:3, 3] = np.random.normal(loc=h[:3, 3], scale=nt)
  return h_noise


# Implements
class Evaluator:
  def __init__(self, exp, env, log=True):
    super().__init__()
    self._log = log
    if self._log:

      files = [str(s) for s in Path(exp["name"]).rglob("*.yml")]

      if env["workstation"]:
        self._run = neptune.init(
          project=exp["neptune_project_name"],
          api_token=os.environ["NEPTUNE_API_TOKEN"],
          tags=[exp["name"], "workstation_" + str(env["workstation"])],
          source_files=files,
        )
      else:
        self._run = neptune.init(
          project=exp["neptune_project_name"],
          api_token=os.environ["NEPTUNE_API_TOKEN"],
          proxies={
            "http": "http://proxy.ethz.ch:3128",
            "https": "http://proxy.ethz.ch:3128",
          },
          tags=[exp["name"], "workstation_" + str(env["workstation"])],
          source_files=files,
        )

    print(exp)
    print(exp["name"])
    print("Flow Checkpoint: ", exp["checkpoint_load"])
    print("Segm Checkpoint: ", exp["checkpoint_load_seg"])
    self._exp = exp
    self._env = env
    self._val = exp.get("val", {})
    self._inferencer = Inferencer(exp, env)
    self.device = "cuda"
    self._inferencer.to(self.device)
    self.iterations = exp["eval_cfg"]["iterations"]

    from visu import Visualizer

    self._visu = Visualizer(
      os.path.join(exp["name"], "visu"), num_classes=2, store=True
    )
    self._visu.logger = self._run
    self.mode = str_to_mode(exp["eval_cfg"]["mode"])  # MUTIPLE_INIT_POSES

  def __del__(self):
    if self._log:
      # Stop logging
      self._run.stop()

  @torch.no_grad()
  def evaluate_full_dataset(self, test_dataloader):
    ycb = test_dataloader.dataset
    if self.mode != Mode.TRACKING:
      ycb.deterministic_random_shuffel()
    ycb.estimate_pose = True
    ycb.err = True
    ycb.valid_flow_minimum = 0
    ycb.fake_flow = not (self._exp["eval_cfg"]["use_gt_valid"] == "gt")
    slow = self._exp["eval_cfg"]["use_gt_valid"] == "gt"

    if self._exp["test_dataset"]["mode"] == "test_tracking":
      sub_sample = 10
    else:
      sub_sample = 1
      elements = len(test_dataloader.dataset._base_path_list)

    if self.mode != Mode.TRACKING:
      tracklets = []
      for i in range(elements):
        tracklets.append([i])
    else:
      paths = np.array(ycb._base_path_list)
      objs = np.array(ycb._obj_idx_list)
      tracklets = compute_tracklets(paths, objs)

    nr_tracklets = len(tracklets)
    tracklet_totals = [len(le) for le in tracklets]
    count = 0
    if self._exp["test_dataset"]["mode"] == "test_tracking":
      elements = np.array(tracklet_totals).sum()
    adds = np.full((elements, self.iterations), np.inf)
    add_s = np.full((elements, self.iterations), np.inf)
    idx_arr = np.full((elements), np.inf)
    epe = np.full((elements, self.iterations), 999)
    init_adds_arr = np.full((elements, self.iterations), np.inf)
    init_add_s_arr = np.full((elements, self.iterations), np.inf)

    h_init_all = np.eye(4)[None, None].repeat(elements, 0).repeat(self.iterations, 1)
    h_pred_all = np.eye(4)[None, None].repeat(elements, 0).repeat(self.iterations, 1)

    ratios_arr = np.zeros((elements, self.iterations))

    r_repro_arr = np.zeros((elements, self.iterations))
    repro_errors = np.full((elements, self.iterations), np.inf)
    valid_corrospondences = np.zeros((elements, self.iterations))

    violation_arr = np.full((elements, self.iterations), Violation.UNKNOWN)

    computed_elements = []
    _st = time.time()
    move_dir = np.zeros((2))
    # ──────────────────────────────────────────────────────────  ─────
    # Iterate over full dataset.
    # ──────────────────────────────────────────────────────────  ─────
    for i, track in enumerate(tracklets):
      print(f"Inferenced Tracklets {i}/{nr_tracklets}")
      valid_element = True
      h_store = None
      # ──────────────────────────────────────────────────────────  ─────
      # Apply network mutiple times.
      # ──────────────────────────────────────────────────────────  ─────
      history_rot = []
      history_trans = []

      track = track[::sub_sample]

      for nr_t, t in enumerate(track):
        computed_elements.append(t)

        for k in range(self.iterations):
          count += 1
          if k == 0 and nr_t == 0 and h_store is None:
            # START TRACKLET AND FIRST ITERATION GET YCB DATA
            batch = ycb.getElement(t)
            h_store = batch[7].detach().cpu().numpy()
          else:
            if self.mode == Mode.REFINEMENT or Mode.TRACKING:
              # LOAD THE STORED POSE ESTIMATE
              current_pose = h_store

            elif self.mode == Mode.MUTIPLE_INIT_POSES:
              # LOAD STORED POSE ESTIMATE WITH TRANSLATION
              h_store[:2, 3] = (
                h_store[:2, 3] + (move_dir.clip(-75, 75) / 75 * 0.04).cpu().numpy()
              )
              current_pose = h_store
            # GET DATA
            batch = ycb.getElement(t, h_real_est=current_pose)

          print(
            f"Tracklet: {nr_t}, ",
            batch[-2][-10:],
            " obj ",
            batch[9],
            "time",
            time.time() - _st,
            "left t:",
            (time.time() - _st) / count * ((sum(tracklet_totals) / sub_sample) - count),
          )

          if batch[0] is None and k == 0:
            print("CANT start given PoseCNN fails!")
            violation_arr[t, k] = Violation.DATALOADER_OBJECT_INVALID
            idx_arr[t] = int(batch[1])

            # Only break if we are not in MUTIPLE INIT POSES mode
            if self.mode != Mode.MUTIPLE_INIT_POSES:
              valid_element = False
              break
            else:
              continue
          else:
            idx_arr[t] = int(batch[9])

          # ACTUAL POSE INFERENCE
          batch = expand_to_batch(batch, self.device)
          flow_predictions, pred_valid = self._inferencer(batch)  # 200ms
          valid_corrospondences[t, k] = int(pred_valid.sum())
          if slow:
            gt_valid = batch[3]
            gt_flow = batch[2]
            _epe = float(
              (
                (
                  torch.sum((flow_predictions[-1] - gt_flow) ** 2, dim=1).sqrt()
                  * gt_valid
                ).sum()
                / gt_valid.sum()
              ).cpu()
            )
          (
            h_gt,
            h_render,
            h_init,
            bb,
            idx,
            K_ren,
            K_real,
            render_d,
            model_points,
            img_real_ori,
            p,
            img_render_ori,
          ) = batch[5:]

          if self._exp["eval_cfg"]["use_gt_valid"] == "gt":
            fv = gt_valid
          elif self._exp["eval_cfg"]["use_gt_valid"] == "pred":
            fv = pred_valid
          elif self._exp["eval_cfg"]["use_gt_valid"] == "none":
            fv = torch.ones_like(pred_valid)

          move_dir = flow_predictions[-1][0, :, fv[0, :, :] == 1].mean(axis=1)

          st = time.time()
          (
            res_dict,
            count_invalid,
            h_pred__pred_pred,
            repro_error,
            ratios,
            valid,
            violations,
          ) = full_pose_estimation(
            h_gt=h_gt,
            h_render=h_render,
            h_init=h_init,
            bb=bb,
            flow_valid=fv,
            flow_pred=flow_predictions[-1],
            idx=idx.clone(),
            K_ren=K_ren,
            K_real=K_real,
            render_d=render_d,
            model_points=model_points,
            cfg=self._exp["eval_cfg"].get("full_pose_estimation", {}),
          )  # 50ms

          if self._env["workstation"] and count < 10:

            bb_real = BoundingBox(bb[0][0], bb[1][0])
            bb_render = BoundingBox(bb[2][0], bb[3][0])
            img_real_crop = bb_real.crop(img_real_ori[0])
            img_render_crop = bb_render.crop(img_render_ori[0])

            img_real_crop = cv2.resize(img_real_crop.cpu().numpy(), (640, 480))
            img_render_crop = cv2.resize(img_render_crop.cpu().numpy(), (640, 480))

            self._visu.epoch = count

            self._visu.plot_image(img_real_ori[0], tag="img_real")
            self._visu.plot_image(img_real_crop, tag="img_real_crop")
            self._visu.plot_image(img_render_ori[0], tag="img_render")
            self._visu.plot_image(img_render_crop, tag="img_render_crop")

            self._visu.plot_detectron(
              img=img_real_crop,
              label=fv[0],
              tag="gt_detectron",
              alpha=0.75,
              text_off=True,
            )
            self._visu.plot_detectron(
              img=img_real_crop,
              label=pred_valid[0],
              tag="pred_detectron",
              alpha=0.75,
              text_off=True,
            )
            self._visu.plot_flow(
              flow_predictions[-1][0].permute(0, 2, 1), tag="pred_flow"
            )
            if slow:
              self._visu.plot_flow(gt_flow[0].permute(0, 2, 1), tag="gt_flow")

              self._visu.plot_corrospondence(
                gt_flow[0, 0, :, :],
                gt_flow[0, 1, :, :],
                fv[0].cpu(),
                torch.tensor(img_real_crop),
                torch.tensor(img_render_crop),
                colorful=False,
                text=False,
                res_h=30,
                res_w=30,
                min_points=50,
                jupyter=False,
                col=(0, 255, 255),
                tag="gt_corro",
              )

            self._visu.plot_corrospondence(
              flow_predictions[-1][0, 0, :, :],
              flow_predictions[-1][0, 1, :, :],
              fv[0].cpu(),
              torch.tensor(img_real_crop),
              torch.tensor(img_render_crop),
              colorful=False,
              text=False,
              res_h=30,
              res_w=30,
              min_points=50,
              jupyter=False,
              col=(0, 255, 255),
              tag="pred_corro",
            )
            b = 0
            img_gt = self._visu.plot_estimated_pose(
              img=img_real_ori[b].cpu().numpy(),
              points=model_points[b].cpu(),
              H=h_gt[b].cpu(),
              K=K_real[b].cpu(),
              color=(0, 255, 255, 255),
              tag="h_gt",
              w=1,
            )
            img_pred = self._visu.plot_estimated_pose(
              img=img_real_ori[b].cpu().numpy(),
              points=model_points[b].cpu(),
              H=h_pred__pred_pred[b].cpu(),
              K=K_real[b].cpu(),
              color=(0, 255, 255, 255),
              tag="h_pred",
              w=1,
            )
            img_init = self._visu.plot_estimated_pose(
              img=img_real_ori[b].cpu().numpy(),
              points=model_points[b].cpu(),
              H=h_init[b].cpu(),
              K=K_real[b].cpu(),
              color=(0, 255, 255, 255),
              tag="h_init",
              w=1,
            )

          ratios_arr[t, k] = ratios[0]
          repro_errors[t, k] = repro_error
          h_init_all[t, k] = h_init.cpu().numpy()[0]
          h_pred_all[t, k] = h_pred__pred_pred.cpu().numpy()[0]
          init_adds_arr[t, k] = res_dict["adds_h_init"]
          init_add_s_arr[t, k] = res_dict["add_s_h_init"]
          r_repro = 0
          if violations[0] == Violation.SUCCESS:
            adds[t, k] = res_dict["adds_h_pred"]
            add_s[t, k] = res_dict["add_s_h_pred"]
            if slow:
              epe[t, k] = _epe

            h_store = h_pred__pred_pred.cpu().numpy()[0]
            patients_count = 0

            pred_p = torch.bmm(
              model_points, torch.transpose(h_pred__pred_pred[:, :3, :3], 1, 2)
            ) + h_pred__pred_pred[:, :3, 3][:, None, :].repeat(
              1, model_points.shape[1], 1
            )
            from ycb.ycb_helper import backproject_points

            points = backproject_points(pred_p[0], K=K_real[0]).type(torch.long).T
            repro = torch.zeros_like(pred_valid)
            points[0, :] = points[0, :].clip(0, repro.shape[1] - 1)
            points[1, :] = points[1, :].clip(0, repro.shape[2] - 1)
            repro[0][points[0, :], points[1, :]] = 1
            bb_real = BoundingBox(bb[0][0], bb[1][0])
            repro_crop = bb_real.crop(
              repro[0][:, :, None].type(torch.float32), scale=True
            )

            footprint = disk(12)

            tmp = binary_dilation(
              repro_crop[:, :, 0].cpu().numpy().astype(np.bool), selem=footprint
            )
            tmp = binary_erosion(tmp, selem=footprint)
            r_sum = (pred_valid.cpu().numpy() * tmp).sum()
            r_repro = r_sum / pred_valid.cpu().numpy().sum()
            r_repro_arr[t, k] = float(r_repro)
            if r_repro < self._exp["eval_cfg"]["reject_ratio"]:
              violations[0] == Violation.FAILED_R_REPRO
              adds[t, k] = res_dict["adds_h_init"]
              add_s[t, k] = res_dict["add_s_h_init"]
              if self.mode == Mode.REFINEMENT:
                violation_arr[t, k] = violations[0]
                break
              patients_count += 1
              # reset to posecnn
              if self.mode == Mode.TRACKING:
                if patients_count > self._exp["eval_cfg"].get("track_patients", 0):
                  h_store = None

            if self._env["workstation"] and count < 10:
              img_real_crop = bb_real.crop(img_real_ori[0])
              img_real_crop = cv2.resize(img_real_crop.cpu().numpy(), (640, 480))

              self._visu.plot_detectron(
                img=img_real_crop,
                label=tmp.astype(np.uint8),
                tag="REPRO Crop",
                alpha=0.75,
                text_off=True,
              )
              self._visu.plot_detectron(
                img=img_real_ori[b].cpu().numpy(),
                label=repro[0].cpu().numpy(),
                tag="REPRO",
                alpha=0.75,
                text_off=True,
              )
              self._visu.plot_detectron(
                img=img_real_crop,
                label=pred_valid[0].cpu().numpy(),
                tag="PREDICTION",
                alpha=0.75,
                text_off=True,
              )
          else:

            adds[t, k] = res_dict["adds_h_init"]
            add_s[t, k] = res_dict["add_s_h_init"]

            if self.mode == Mode.REFINEMENT:
              violation_arr[t, k] = violations[0]
              break

            patients_count += 1
            # reset to posecnn
            if self.mode == Mode.TRACKING:
              if patients_count > self._exp["eval_cfg"].get("track_patients", 0):
                h_store = None

          if h_store is None:
            history_rot = []
            history_trans = []
          else:
            history_trans.append(
              np.linalg.norm(h_init.cpu().numpy()[0, :3, 3] - h_store[:3, 3])
            )
            if len(history_trans) > 10:
              history_trans = history_trans[1:]

            if np.array(history_trans).mean() > self._exp["eval_cfg"].get(
              "trans_difference", 0.02
            ):
              print("RESET BASED ON TRANS")
              violations[0] = Violation.TRANS_DIFFERENCE
              history_rot = []
              history_trans = []
              h_store = None

            self._run["trans_mean"].log(np.array(history_trans).mean())

          violation_arr[t, k] = violations[0]
        # ───  ────────────────────────────────────────────────────────────
        #      ONLY LOGGING
        # ───  ────────────────────────────────────────────────────────────
        self._run["count"].log(count)

        if count % 100 == 0 and count != 0:
          print("PROGRESS REPORT COUNT, ", count)
          mask = np.array(computed_elements)
          add_s_finite = np.isfinite(add_s[mask])
          sm = add_s_finite.sum(axis=1) - 1
          sm[sm < 0] = 0
          sel = np.eye(self.iterations)[sm] == 1

          print(
            f"final after {self.iterations}th-iteration: ",
            compute_auc(add_s[mask][sel]),
          )
          print("Mean 1th-iteration: ", compute_auc(add_s[mask, 0]))
          print(
            "AUC best over all iterations: ",
            compute_auc(np.min(add_s[mask, :], axis=1)),
          )
          tar = np.argmax(ratios_arr[mask], axis=1)
          sel = np.zeros_like(ratios_arr[mask])
          for _j, _i in enumerate(tar.tolist()):
            sel[_j, _i] = 1
          sel = sel == 1
          print("Best RANSAC ratios: ", compute_auc(add_s[mask][sel]))

          sel2 = np.argmin(valid_corrospondences[mask], axis=1)
          sel2 = np.eye(valid_corrospondences.shape[1])[sel2] == 1
          print("AUC best valids: ", compute_auc(add_s[mask][sel2]))
          print("INIT ADDS PoseCNN: ", compute_auc(init_add_s_arr[mask][:, 0]))

        # STOING INTERMEDATE RESULTS PICKLE
        if count % 5000 == 0:
          st = self._exp["eval_cfg"]["output_filename"]
          b = os.path.join(self._exp["name"], f"{self.mode}_{st}_data_{count}.pkl")
          dic = {
            "add_s": add_s,
            "adds": adds,
            "idx_arr": idx_arr,
            "ratios_arr": ratios_arr,
            "valid_corrospondences": valid_corrospondences,
            "init_adds_arr": init_adds_arr,
            "init_add_s_arr": init_add_s_arr,
            "epe": epe,
            "h_init_all": h_init_all,
            "h_pred_all": h_pred_all,
            "violation_arr": violation_arr,
            "repro_errors": repro_errors,
            "mask": np.array(computed_elements),
            "r_repro_arr": r_repro_arr,
          }
          import pickle

          with open(b, "wb") as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
          self._run[f"result_inter_{i}"].upload(b)

        # NEPTUNE LOGGING
        if self._log:
          mask = np.array(computed_elements)
          logs = {
            "add_s": add_s,
            "adds": adds,
            "ratios_arr": ratios_arr,
            "valid_corrospondences": valid_corrospondences,
            "epe": epe,
            "init_adds_arr": init_adds_arr,
            "init_add_s_arr": init_add_s_arr,
          }
          for k, v in logs.items():
            for iter in range(self.iterations):
              self._run[k + f"_iter_{iter}"].log(v[t, iter])

          logs = {"idx_arr": idx_arr}

          self._run["r_repro"].log(r_repro)
          for k, v in logs.items():
            self._run[k + f"_iter"].log(v[t])

          if count % 10 == 0 and count != 0:
            # compute aucs
            for iter in range(self.iterations):
              self._run["auc_add_s" + f"_iter_{iter}"].log(
                compute_auc(add_s[mask, iter])
              )
              self._run["auc_adds" + f"_iter_{iter}"].log(compute_auc(adds[mask, iter]))
              self._run["auc_init_adds" + f"_iter_{iter}"].log(
                compute_auc(init_adds_arr[mask, iter])
              )
              self._run["auc_init_add_s" + f"_iter_{iter}"].log(
                compute_auc(init_add_s_arr[mask, iter])
              )

            for _j in range(21):
              m = idx_arr[mask] == _j
              for iter in range(self.iterations):
                self._run[f"auc_add_s_obj_{_j}" + f"_iter_{iter}"].log(
                  compute_auc(add_s[mask][m, iter])
                )
                self._run[f"auc_adds_obj_{_j}" + f"_iter_{iter}"].log(
                  compute_auc(adds[mask][m, iter])
                )

                self._run[f"auc_init_adds_obj_{_j}" + f"_iter_{iter}"].log(
                  compute_auc(init_adds_arr[mask][m, iter])
                )
                self._run[f"auc_init_add_s_obj_{_j}" + f"_iter_{iter}"].log(
                  compute_auc(init_add_s_arr[mask][m, iter])
                )

    # STOING FINAL RESULTS PICKLE
    st = self._exp["eval_cfg"]["output_filename"]
    b = os.path.join(self._exp["name"], f"{self.mode}_{st}_data_final.pkl")
    dic = {
      "add_s": add_s,
      "adds": adds,
      "idx_arr": idx_arr,
      "ratios_arr": ratios_arr,
      "valid_corrospondences": valid_corrospondences,
      "init_adds_arr": init_adds_arr,
      "init_add_s_arr": init_add_s_arr,
      "epe": epe,
      "h_init_all": h_init_all,
      "h_pred_all": h_pred_all,
      "violation_arr": violation_arr,
      "repro_errors": repro_errors,
      "r_repro_arr": r_repro_arr,
    }
    import pickle

    varss = np.array([a.value for a in violation_arr[:, 0]])
    print(np.unique(varss, return_counts=True))
    with open(b, "wb") as handle:
      pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    self._run["result_final"].upload(b)

    sym = []
    for ind in idx_arr.tolist():
      sym.append(
        not (int(ind) + 1 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18])
      )
    sym = np.array(sym)
    for i in range(self.iterations):
      non_sym = sym == False
      mix = adds[sym, i].tolist() + add_s[non_sym, i].tolist()
      self._run[f"auc_s_mix_iter_{i}"].log(compute_auc(np.array(mix)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--exp",
    type=file_path,
    default="cfg/exp/final/1_pose_prediction/pose_estimation.yml",
    help="The main experiment yaml file.",
  )

  args = parser.parse_args()
  exp_cfg_path = args.exp
  env_cfg_path = os.path.join("cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yml")

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

  if exp.get("timestamp", True):
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    model_path = os.path.join(env["base"], exp["name"])
    p = model_path.split("/")
    model_path = os.path.join("/", *p[:-1], str(timestamp) + "_" + p[-1])
  else:
    model_path = os.path.join(env["base"], exp["name"])
    shutil.rmtree(model_path, ignore_errors=True)

  # Create the directory
  Path(model_path).mkdir(parents=True, exist_ok=True)

  # Only copy config files for the main ddp-task
  exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
  env_cfg_fn = os.path.split(env_cfg_path)[-1]
  print(f"Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}")
  shutil.copy(exp_cfg_path, f"{model_path}/{exp_cfg_fn}")
  shutil.copy(env_cfg_path, f"{model_path}/{env_cfg_fn}")
  exp["name"] = model_path

  inference_manager = Evaluator(exp=exp, env=env)

  # LOAD WEIGHTS
  p = os.path.join(env["base"], exp["checkpoint_load"])
  if os.path.isfile(p):
    res = torch.load(p)
    out = inference_manager._inferencer.load_state_dict(res["state_dict"], strict=False)

    if len(out[1]) > 0:
      print("Restore weights from ckpts", out)
      raise Exception(f"Not found seg checkpoint: {p}")
    else:
      print("Restore flow-weights from ckpts successfull")
  else:
    raise Exception(f"Not found flow checkpoint: {p}")
  p = os.path.join(env["base"], exp["checkpoint_load_seg"])
  if os.path.isfile(p):
    res = torch.load(p)
    new_statedict = {}
    for (k, v) in res["state_dict"].items():
      new_statedict[k.replace("model", "seg")] = v
    out = inference_manager._inferencer.load_state_dict(new_statedict, strict=False)

    if len(out[1]) > 0:
      print("Restore_seg weights from ckpts", out)
      raise Exception(f"Not found seg checkpoint: {p}")
    else:
      print("Restore seg-weights from ckpts successfull")
  else:
    raise Exception(f"Not found seg checkpoint: {p}")

  # PERFORME EVALUATION
  test_dataloader = datasets.fetch_dataloader(exp["test_dataset"], env)
  inference_manager.evaluate_full_dataset(test_dataloader)
