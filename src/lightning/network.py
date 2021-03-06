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
from src_utils import DotDict
from raft import RAFT
__all__ = ['Network']

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics
    
class Network(LightningModule):
  def __init__(self, exp, env):
    super().__init__()
    self._exp = exp
    self._env = env
    self.hparams['lr'] = self._exp['lr']
    
    
    self.model = RAFT(args = DotDict(self._exp['model']['args']) )
    
    self._mode = 'train'
    self._logged_images = {'train': 0, 'val': 0, 'test': 0}
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    
    
  def forward(self, batch, **kwargs):
    image1 = batch[0]
    image2 = batch[1]
    flow_predictions = self.model(image1, image2, iters=self._exp['model']['iters'])
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
    """
    img1 0-255 BS,C,H,W
    img2 0-255 BS,C,H,W
    flow BS,2,H,W   max [-155 263]
    valid 0 or 1 

    flow_predictons is a list len(flow_predictions) = iters , flow_predictions[0].shape == flow.shape 
    """
    BS = batch[0].shape[0] 
    flow = batch[2]
    valid = batch[3]
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
  
  