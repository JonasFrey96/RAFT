import torch.utils.data as data
from .ycb import YCB

__all__ = "get_ycb_dataloader"


def get_ycb_dataloader(cfg, env):
  train_dataset = YCB(
    root=env["ycb"],
    mode=cfg["mode"],
    image_size=cfg["image_size"],
    cfg_d=cfg["cfg_ycb"],
  )
  train_loader = data.DataLoader(train_dataset, **cfg["loader"], drop_last=True)
  return train_loader
