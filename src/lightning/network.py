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

#from .metrices import IoU, PixAcc
from visu import Visualizer
import datetime
from math import ceil
from models import RPOSE


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
    
    
		self.model = FastSCNN(**self._exp['model']['cfg'])
    
    p_visu = os.path.join( self._exp['name'], 'visu')
    self.visualizer = Visualizer(
      p_visu=p_visu,
      logger=None,
      num_classes=self._exp['model']['cfg']['num_classes']+1)
    
    self._mode = 'train'
    
    self.test_acc = pl_metrics.classification.Accuracy()
    self.val_acc = pl_metrics.classification.Accuracy()


    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0
    
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    
    
  def forward(self, batch, **kwargs):
		outputs = self.model(batch)
    return outputs
  
  def on_train_epoch_start(self):
    self._mode = 'train'
     
  def on_train_start(self):
    print('Start')
    self.visualizer.logger= self.logger
    
  def on_epoch_start(self):
    self.visualizer.epoch = self.current_epoch
    self.logged_images_train = 0
    self.logged_images_val = 0
    self.logged_images_test = 0
          
  def compute_loss(self, pred, target):
		loss = F.cross_entropy(pred, target, ignore_index=-1)
    return loss
  
  def training_step(self, batch, batch_idx):
    images = batch[0]
    target = batch[1]
    BS = images.shape[0]

		outputs = self(batch = images)

    loss = self.compute_loss(  
              pred = outputs[0], 
              target = target)
        
    self.log('train_loss', loss, on_step=False, on_epoch=True)
    return {'loss': loss, 'pred': outputs[0], 'target': target}
  
  def training_step_end(self, outputs):
    # Log replay buffer stats
    self.logger.log_metrics( 
      metrics = { 'real': torch.tensor(self._real_samples),
                  'replayed': torch.tensor(self._replayed_samples)},
      step = self.global_step)
  
    # Logging + Visu
    self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True, prog_bar = True)
    
    if ( self._exp['visu'].get('train_images',0) > self.logged_images_train and 
         self.current_epoch % self._exp['visu'].get('every_n_epochs',1) == 0):
			self.logged_images_val += 1
      self.visualizer.plot()
      
    return {'loss': outputs['loss']}
        
  def on_train_epoch_end(self, outputs):
    if self.current_epoch % self.trainer.check_val_every_n_epoch != 0:
      self.log('val_loss',val_loss)

  def validation_step(self, batch, batch_idx, dataloader_idx=0):      
    images = batch[0]
    target = batch[1]
    outputs = self(images)
    loss = F.cross_entropy(outputs[0], target, ignore_index=-1 ) 
    pred = torch.argmax(outputs[0], 1)
    
    return {'pred': pred, 'target': target, 'ori_img': batch[2], 'dataloader_idx': dataloader_idx, 'loss_ret': loss }

  def validation_step_end( self, outputs ):
    # Logging + Visu
    if ( self._exp['visu'].get('val_images',0) > self.logged_images_val and 
         self.current_epoch % self._exp['visu'].get('every_n_epochs',1)== 0) :
      self.logged_images_val += 1
      self.visualizer.plot()
      
    self.log(f'val_acc', self.val_acc[dataloader_idx] , on_epoch=True, prog_bar=False)
    self.log('val_loss', outputs['loss_ret'], on_epoch=True)
  
  def on_validation_epoch_start(self):
    self._mode = 'val'
  
  def validation_epoch_end(self, outputs):
    metrics = self.trainer.logger_connector.callback_metrics
   
    epoch = str(self.current_epoch)
    t = time.time()- self._epoch_start_time
    t = str(datetime.timedelta(seconds=round(t)))
    t2 = time.time()- self._train_start_time
    t2 = str(datetime.timedelta(seconds=round(t2))) 
    if not self.trainer.running_sanity_check:
      print('VALIDATION_EPOCH_END: Time for a complete epoch: '+ t)
      n = self._task_name
      n = wrap(n,20)
      t = wrap(t,10,True)
      epoch =  wrap(epoch,3)
      t_l = wrap(t_l,6)
      v_acc = wrap(v_acc,6)
      v_mIoU = wrap(v_mIoU,6)
      
      print('VALIDATION_EPOCH_END: '+ 
        f"Exp: {n} | Epoch: {epoch} | TimeEpoch: {t} | TimeStart: {t2} |  >>> Train-Loss: {t_l } <<<   >>> Val-Acc: {v_acc}, Val-mIoU: {v_mIoU} <<<"
      )
    self._epoch_start_time = time.time()


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
  
  