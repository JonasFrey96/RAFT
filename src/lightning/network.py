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

import datetime
from math import ceil
from models import RPOSE


import RAFT
__all__ = ['Network']

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

class Network(LightningModule):
  def __init__(self, exp, env):
    super().__init__()
    self._exp = exp
    self._env = env
    self.hparams['lr'] = self._exp['lr']
    
    
		self._model = RAFT(**self._exp['model']['cfg'])
    
    self._mode = 'train'
    self._logged_images = {'train': 0, 'val': 0, 'test': 0}
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    
    
  def forward(self, batch, **kwargs):
    image1 = batch[0]
    image2 = batch[1]
		flow_predictions = self._model(image1, image2, iters=self._exp['model']['iters'])
    return flow_predictions
  
  def on_train_epoch_start(self):
    self._mode = 'train'
     
  def on_train_start(self):
    pass

  def on_epoch_start(self):
    # RESET IMAGE COUNT
    for k in self._logged_images.keys():
      self._logged_images[k] = 0
          
  def training_step(self, batch, batch_idx):
    BS = batch[0].shape[0]
    flow = batch[2]
    valid = valid[3]
		flow_predictions = self(batch = batch)
    loss, metrics = sequence_loss(flow_predictions, flow, valid, self._exp['model']['gamma'])
    return {'loss': loss, 'pred': flow_predictions, 'target': flow}
  
  def training_step_end(self, outputs):
    # Log replay buffer stats
    self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True)
    return {'loss': outputs['loss']}
        
  def validation_step(self, batch, batch_idx, dataloader_idx=0):      
    return self.training_step(batch, batch_idx)
  
  def validation_step_end( self, outputs ):
    self.log('val_loss', outputs['loss'], on_step=False, on_epoch=True)
  
  def on_validation_epoch_start(self):
    self._mode = 'val'
  
  def validation_epoch_end(self, outputs):
    pass

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
  
  