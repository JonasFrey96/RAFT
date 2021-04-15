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
from models_asl import FastSCNN, Teacher, ReplayStateSyncBack
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
    self.model = FastSCNN(**self._exp['model']['cfg'])
    
    p_visu = os.path.join( self._exp['name'], 'visu')
    
    self.visualizer = Visualizer(
      p_visu=p_visu,
      logger=None,
      num_classes=self._exp['model']['cfg']['num_classes']+1)
    self._mode = 'train'
  
  def forward(self, batch, **kwargs):
    
    return self.model(batch)
  
  def on_train_epoch_start(self):
    self._mode = 'train'
     
  def on_train_start(self):
    print('Start')
    self.visualizer.logger= self.logger
    
  def on_epoch_start(self):
    self.visualizer.epoch = self.current_epoch
  
  
  def training_step(self, batch, batch_idx):
    real = batch[0]
    render = batch[1]
    target = batch[2]
    synthetic = batch[3]
    BS, C, H, W = real.shape
    inp = torch.cat ( [real,render],dim=1)
    
    outputs = self(batch = inp )
    loss =F.cross_entropy(outputs[0], target, ignore_index=-1, reduction='none').mean(dim=(1,2))      
    pred = torch.argmax(outputs[0], 1)

    acc = (pred == target).sum() / (BS * H * W)  
    
    self.log(f'{self._mode}_acc', acc, on_step=False, on_epoch=True)
    self.log(f'{self._mode}_loss', loss.mean(), on_step=False, on_epoch=True)

    if synthetic.sum() > 0:
      self.log(f'{self._mode}_render_loss', loss[synthetic].mean(), on_step=False, on_epoch=True)

    non_synthetic = synthetic == False
    if non_synthetic.sum() > 0:
      self.log(f'{self._mode}_real_loss', loss[non_synthetic].mean(), on_step=False, on_epoch=True)

    loss = loss.mean()
    return {'loss': loss }
  
  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    return self.training_step(batch, batch_idx)
  
  def on_test_epoch_start(self):
    self._mode = 'val'

  def on_test_epoch_start(self):
    self._mode = 'test'
    
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
    else:
      raise Exception

    if self._exp.get('lr_scheduler',{}).get('active', False):
      #polynomial lr-scheduler
      init_lr = self.hparams['lr']
      max_epochs = self._exp['lr_scheduler']['cfg']['max_epochs'] 
      target_lr = self._exp['lr_scheduler']['cfg']['target_lr'] 
      power = self._exp['lr_scheduler']['cfg']['power'] 
      lambda_lr= lambda epoch: (((max_epochs-min(max_epochs,epoch) )/max_epochs)**(power) ) + (1-(((max_epochs -min(max_epochs,epoch))/max_epochs)**(power)))*target_lr/init_lr
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch=-1, verbose=True)
      ret = [optimizer], [scheduler]
    else:
      ret = [optimizer]
    return ret