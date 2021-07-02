# STD
import copy
import sys
import os
import time
import shutil
import argparse
import logging
import signal
import pickle
import math
from pathlib import Path
import random 
from math import pi
from math import ceil
import logging

# MISC 
import numpy as np
import pandas as pd

# DL-framework
import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning import metrics as pl_metrics
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from torchvision.utils import make_grid
from torch.nn import functional as F
# MODULES
from models_asl import FastSCNN
from visu import Visualizer

#from .metrices import IoU, PixAcc
import datetime
from math import ceil

__all__ = ['Network']
def wrap(s,length, hard=False):
  if len(s) < length:
    return s + ' '*(length - len(s))
  if len(s) > length and hard:
    return s[:length]
  return s

class Network(LightningModule):
  def __init__(self, exp, env):
    super().__init__()
    self._epoch_start_time = time.time()
    self._exp = exp
    self._env = env
    self.hparams['lr'] = self._exp['lr']
    print(self._exp)
    self.model = FastSCNN(**self._exp['seg']['cfg'])
    
    p_visu = os.path.join( self._exp['name'], 'visu')
    self._output_transform = transforms.Compose([
          transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    self.visualizer = Visualizer(
      p_visu=p_visu,
      logger=None,
      num_classes=self._exp['seg']['cfg']['num_classes']+1)
    self._mode = 'train'

    self._plot_images = {'train': 0, 'val':0, 'test':0} 
    self._plot_images_max = {'train': 3, 'val': 3, 'test': 3}

  def forward(self, batch, **kwargs):
    return self.model(batch)
  
  def on_train_epoch_start(self):
    self._mode = 'train'
    for k in self._plot_images.keys():
      self._plot_images[k] = 0
    
  def on_train_start(self):
    print('Start')
    self.visualizer.logger= self.logger
    
  def on_epoch_start(self):
    self.visualizer.epoch = self.current_epoch
  
  def training_step(self, batch, batch_idx):
    real = self._output_transform(batch[0])
    render = self._output_transform(batch[1])
    target = batch[2]
    synthetic = batch[3]
    
    BS, C, H, W = real.shape
    inp = torch.cat ( [real,render],dim=1)
    
    outputs = self(batch = inp )
    loss =F.cross_entropy(outputs[0], target, ignore_index=-1, reduction='none').mean(dim=(1,2))      
    pred = torch.argmax(outputs[0], 1)

    # LOG    
    self.plot(batch[0], batch[1], pred, target)

    # COMPUTE STATISTICS
    
    acc = (pred == target).sum() / (BS * H * W)  
    
    TN = ((pred == 0) * (target == 0)).sum().float()
    FP = ((pred == 1) * (target == 0)).sum().float()
    TP = ((pred == 1) * (target == 1)).sum().float() 
    FN = ((pred == 0) * (target == 1)).sum().float()
    s = (TN + FP +TP + FN).float() 
    TN /= s
    FP /= s
    TP /= s
    FN /= s
    self.log(f'{self._mode}_TN_ratio', TN, on_step=False, on_epoch=True)
    self.log(f'{self._mode}_TP_ratio', TP, on_step=False, on_epoch=True)
    self.log(f'{self._mode}_FN_ratio', FN, on_step=False, on_epoch=True)
    self.log(f'{self._mode}_FP_ratio', FP, on_step=False, on_epoch=True)

    self.log(f'{self._mode}_acc', acc, on_step=False, on_epoch=True)
    self.log(f'{self._mode}_loss', loss.mean(), on_step=False, on_epoch=True)

    if synthetic.sum() > 0:
      self.log(f'{self._mode}_render_loss', loss[synthetic].mean(), on_step=False, on_epoch=True)

    non_synthetic = synthetic == False
    if non_synthetic.sum() > 0:
      self.log(f'{self._mode}_real_loss', loss[non_synthetic].mean(), on_step=False, on_epoch=True)

    loss = loss.mean()
    return {'loss': loss }
  
  def plot(self, ori_real, ori_render, pred, target):
    i = int(self._plot_images[self._mode])
    self.visualizer.plot_image( tag="abc",  img = np.uint8(  np.random.randint(0,255,(100,100,3)) ), method='default')

    if self._plot_images[self._mode] < self._plot_images_max[self._mode] :
      self._plot_images[self._mode] += 1
      print("PERFORM PLOT")
      BS = pred.shape[0]
      rows = int( BS**0.5 )
      grid_target = make_grid(target[:,None].repeat(1,3,1,1),nrow = rows, padding = 2,
              scale_each = False, pad_value = 2)
      grid_pred = make_grid(pred[:,None].repeat(1,3,1,1),nrow = rows, padding = 2,
              scale_each = False, pad_value = 2)

      grid_ori_real = make_grid(ori_real,nrow = rows, padding = 2,
              scale_each = False, pad_value = 0)
      grid_ori_render = make_grid(ori_render,nrow = rows, padding = 2,
              scale_each = False, pad_value = 0)

      self.visualizer.plot_segmentation( label = grid_target[0], method= 'right')
      self.visualizer.plot_segmentation( label = grid_pred[0], method= 'left', tag=f"{self._mode}_Left_Pred__GT_right_{i}")

      self.visualizer.plot_image( img = grid_ori_real, method= 'right')
      self.visualizer.plot_segmentation( label = grid_pred[0], method= 'left', tag=f"{self._mode}_Left_Pred__Right_Image_{i}")

      self.visualizer.plot_image( img = torch.cat( [grid_ori_real, grid_ori_render], dim=2) , method= 'right')
      self.visualizer.plot_segmentation( label = grid_pred[0], method= 'left', tag=f"{self._mode}_Left_Pred__Right_Composed-Image_{i}")

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    return self.training_step(batch, batch_idx)
  
  def on_validation_epoch_start(self):
    self._mode = 'val'
    for k in self._plot_images.keys():
      self._plot_images[k] = 0
    

  def on_test_epoch_start(self):
    self._mode = 'test'
    for k in self._plot_images.keys():
      self._plot_images[k] = 0
    
  def test_step(self, batch, batch_idx):
    return self.training_step(batch, batch_idx)
  
  def configure_optimizers(self):
    if self._exp['optimizer']['name'] == 'ADAM':
      optimizer = torch.optim.Adam(
          [{'params': self.model.parameters()}], lr=self.hparams['lr'])
    elif self._exp['optimizer']['name'] == 'SGD':
      optimizer = torch.optim.SGD(
          [{'params': self.model.parameters()}], lr=self.hparams['lr'],
          **self._exp['optimizer']['sgd_cfg'] )
    elif self._exp['optimizer']['name'] == 'WADAM':
      optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams['lr'],**self._exp['optimizer']['wadam_cfg'] )
    else:
      raise Exception

    if self._exp.get('lr_scheduler',{}).get('active', False):
      if self._exp['lr_scheduler']['name'] == 'POLY':
        #polynomial lr-scheduler
        init_lr = self.hparams['lr']
        max_epochs = self._exp['lr_scheduler']['poly_cfg']['max_epochs'] 
        target_lr = self._exp['lr_scheduler']['poly_cfg']['target_lr'] 
        power = self._exp['lr_scheduler']['poly_cfg']['power'] 
        lambda_lr= lambda epoch: (((max_epochs-min(max_epochs,epoch) )/max_epochs)**(power) ) + (1-(((max_epochs -min(max_epochs,epoch))/max_epochs)**(power)))*target_lr/init_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1, verbose=True)
      elif self._exp['lr_scheduler']['name'] == 'OneCycleLR':
        num_steps = self._exp['lr_scheduler']['onecyclelr_cfg']['num_steps']

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.hparams['lr'], total_steps = num_steps+100,
          pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
      
      lr_scheduler = {
                    'scheduler': scheduler,
                    'interval': "step" }

      ret = {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    else:
      ret = [optimizer]
    return ret