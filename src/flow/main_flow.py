import os
import sys

sys.path.insert(0, os.getcwd())

import coloredlogs

coloredlogs.install()

import shutil
import datetime
import argparse
import signal
import yaml
from pathlib import Path
import copy

# Frameworks
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler

# Costume Modules
from src.common.utils import (
  file_path,
  load_yaml,
  get_neptune_logger,
  get_tensorboard_logger,
)
from lightning import Network
from src.flow.callbacks import TimeCallback
from src.flow.raft import fetch_dataloader


if __name__ == "__main__":

  def signal_handler(signal, frame):
    print("exiting on CRTL-C")
    sys.exit(0)

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--exp",
    type=file_path,
    default="cfg/exp/final/0_train_flow_24/sgd/sgd_medium.yml",
    help="The main experiment yaml file.",
  )

  args = parser.parse_args()
  exp_cfg_path = args.exp
  env_cfg_path = os.path.join("cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yml")

  seed_everything(42)
  local_rank = int(os.environ.get("LOCAL_RANK", 0))

  if local_rank != 0:
    print(local_rank)
    rm = exp_cfg_path.find("cfg/exp/") + len("cfg/exp/")
    exp_cfg_path = os.path.join(exp_cfg_path[:rm], "tmp/", exp_cfg_path[rm:])

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

  # Move YCB Dataset
  if env["workstation"] is False and type(exp["move_datasets"]) == list:
    print("Moveing datasets")
    for dataset in exp["move_datasets"]:
      print(f"Move {dataset}")
      if local_rank == 0:
        print("Start Moveing dataset")
        os.system(
          f"/cluster/home/jonfrey/miniconda3/envs/track4/bin/python scripts/move_datasets.py --datasets={dataset}"
        )
      env[dataset] = os.path.join(os.environ.get("TMPDIR"), dataset)

  if local_rank == 0:
    # Set in name the correct model path
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
  else:
    # the correct model path has already been written to the yaml file.
    model_path = os.path.join(exp["name"], f"rank_{local_rank}")
    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

  # SET GPUS
  if (exp["trainer"]).get("gpus", -1) == -1 and os.environ[
    "ENV_WORKSTATION_NAME"
  ] != "hyrax":
    nr = torch.cuda.device_count()
    print(f"Set GPU Count for Trainer to {nr}!")
    for i in range(nr):
      print(f"Device {i}: ", torch.cuda.get_device_name(i))
    exp["trainer"]["gpus"] = nr

  model = Network(exp=exp, env=env)

  lr_monitor = LearningRateMonitor(**exp["lr_monitor"]["cfg"])

  print(exp)
  if exp["cb_early_stopping"]["active"]:
    early_stop_callback = EarlyStopping(**exp["cb_early_stopping"]["cfg"])
    cb_ls = [early_stop_callback, lr_monitor]
  else:
    cb_ls = [lr_monitor]

  if exp["cb_checkpoint"]["active"]:
    m = "/".join([a for a in model_path.split("/") if a.find("rank") == -1])
    checkpoint_callback = ModelCheckpoint(
      dirpath=m,
      filename="Checkpoint-{epoch:02d}--{step:06d}",
      **exp["cb_checkpoint"]["cfg"],
    )
    cb_ls.append(checkpoint_callback)

  if not exp.get("offline_mode", False):
    logger = get_neptune_logger(
      exp=exp,
      env=env,
      exp_p=exp_cfg_path,
      env_p=env_cfg_path,
      project_name=exp["neptune_project_name"],
    )
    exp["experiment_id"] = logger.experiment.id
    print("Created Experiment ID: " + str(exp["experiment_id"]))
  else:
    logger = get_tensorboard_logger(
      exp=exp, env=env, exp_p=exp_cfg_path, env_p=env_cfg_path
    )

  if exp["cb_time"]["active"]:
    cb_ls.append(TimeCallback(**exp["cb_time"].get("cfg", {})))

  # WRITE BACK NEW CONF FOR OTHER DDPs
  if local_rank == 0:
    rm = exp_cfg_path.find("cfg/exp/") + len("cfg/exp/")
    exp_cfg_path = os.path.join(exp_cfg_path[:rm], "tmp/", exp_cfg_path[rm:])
    Path(exp_cfg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(exp_cfg_path, "w+") as f:
      yaml.dump(exp, f, default_flow_style=False, sort_keys=False)

  # PROFILER
  if exp["trainer"].get("profiler", False):
    exp["trainer"]["profiler"] = AdvancedProfiler(
      output_filename=os.path.join(model_path, "profile.out")
    )
  else:
    exp["trainer"]["profiler"] = False

  if env["workstation"]:

    for n in ["test", "val", "train"]:
      exp[n + "_dataset"]["loader"]["num_workers"] = 4

  # TRAINER
  if exp.get("checkpoint_restore", False):
    p = os.path.join(env["base"], exp["checkpoint_load"])

    trainer = Trainer(
      **exp["trainer"],
      default_root_dir=model_path,
      callbacks=cb_ls,
      resume_from_checkpoint=p,
      logger=logger,
    )
  else:
    trainer = Trainer(
      **exp["trainer"], default_root_dir=model_path, callbacks=cb_ls, logger=logger
    )
    # WEIGHTS

  if exp.get("weights_restore", False):
    p = os.path.join(env["base"], exp["checkpoint_load"])
    if os.path.isfile(p):
      res = torch.load(p)

      if p.find(".pth") != -1:
        msd = {k.replace("module", "model"): v for k, v in res.items()}
      else:
        msd = res["state_dict"]
      out = model.load_state_dict(msd, strict=False)
      print("Restoere weights from ckpts", out)

  if exp.get("weights_restore_seg", False) and model._estimate_pose == True:
    p = os.path.join(env["base"], exp["checkpoint_load_seg"])
    if os.path.isfile(p):
      res = torch.load(p)

      new_statedict = {}
      for (k, v) in res["state_dict"].items():
        new_statedict[k.replace("model", "seg")] = v

      out = model.load_state_dict(new_statedict, strict=False)
      print("Restore_seg weights from ckpts", out)

  if exp.get("mode", "train").find("train") != -1:
    train_dataloader = fetch_dataloader(exp["train_dataset"], env)
    val_dataloader = fetch_dataloader(exp["val_dataset"], env)
    val_dataloader.dataset.deterministic_random_shuffel()
    train_res = trainer.fit(
      model=model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
    )

  if exp.get("mode", "train").find("test") != -1:
    # exp['weights_restore_seg'] = True
    exp_eval = copy.deepcopy(exp)
    exp_eval["weights_restore"] = True

    cl = exp_eval["name"] + "last.ckpt"
    cl[cl.find(env["base"]) + len(env["base"]) :]
    exp_eval["checkpoint_load"] = cl
    exp_eval["mode"] = "test"
    from lightning import Evaluator

    torch.cuda.empty_cache()
    inference_manager = Evaluator(exp_eval, env)

    test_dataloader = fetch_dataloader(exp["test_dataset"], env)

    inference_manager._inferencer.load_state_dict(model.state_dict(), strict=False)
    # test_dataloader.dataset.deterministic_random_shuffel()
    p = os.path.join(env["base"], exp["checkpoint_load_seg"])
    if os.path.isfile(p):
      res = torch.load(p)
      new_statedict = {}
      for (k, v) in res["state_dict"].items():
        new_statedict[k.replace("model", "seg")] = v
      out = inference_manager._inferencer.load_state_dict(new_statedict, strict=False)
      print("Restore_seg weights from ckpts", out)

    inference_manager.evaluate_full_dataset(test_dataloader)

    # trainer.test(model = model,
    #     test_dataloaders = test_dataloader,
    #     ckpt_path =os.path.join( env['base'],exp['checkpoint_load']) )

  try:
    logger.experiment.stop()
  except:
    pass
