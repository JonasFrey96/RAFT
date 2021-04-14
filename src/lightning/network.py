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
from torch import from_numpy as fn
# MODULES

import datetime
from math import ceil
from src_utils import DotDict
from raft import RAFT
from visu import Visualizer
from pose_estimation import full_pose_estimation, compute_auc, compute_percentage
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
    self._logged_images_max = {'train': 2, 'val': 2, 'test': 50}
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    self._visu = Visualizer( os.path.join ( exp['name'], "visu") )

    if self._exp.get('mode','train') == 'test':
      self._estimate_pose = True
    else:
      self._estimate_pose = False
      
    # p_visu, writer=None, num_classes=20, epoch=0, store=True
    
  def forward(self, batch, **kwargs):
    image1 = batch[0]
    image2 = batch[1]
    flow_predictions = self.model(image1, image2, iters=self._exp['model']['iters'])
    
    self.plot( batch[2], flow_predictions, image1, image2, batch[3])
    return flow_predictions
  
  def on_train_epoch_start(self):
    self._visu.logger= self.logger
    self._mode = 'train'
     
  def on_train_start(self):
    pass

  def on_epoch_start(self):
    # RESET IMAGE COUNT
    for k in self._logged_images.keys():
      self._logged_images[k] = 0
    self._visu.epoch = self.trainer.current_epoch
    self.log ( "current_epoch", self.trainer.current_epoch )
    self.log ( "gloabal_step", self.trainer.global_step )
          
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

    if self._estimate_pose:
      h_gt, h_render, h_init, bb, idx, K_ren, K_real, render_d, model_points = batch[4:]
      res_dict, count_invalid = full_pose_estimation( 
        h_gt = h_gt.clone(), 
        h_render = h_render.clone(),
        h_init = h_init.clone(),
        bb = bb, 
        flow_valid = valid.clone(), 
        flow_pred = flow_predictions[-1].clone(), 
        flow_gt = flow.clone(),
        idx = idx.clone(),
        K_ren = K_ren,
        K_real = K_real,
        render_d = render_d.clone(),
        model_points = model_points.clone(),
        cfg = self._exp.get("full_pose_estimation", {})
      )
      try:
        self.count_suc += BS-count_invalid
        self.count_failed += count_invalid
      except:
        self.count_suc = BS-count_invalid
        self.count_failed = count_invalid
      if len( res_dict ) > 0: 
        index_key = str( int( idx ))
        self._adds[index_key].append( float(res_dict["adds_h_pred"].cpu()) )
        self._add_s[index_key].append( float(res_dict["add_s_h_pred"].cpu()) )

        self._adds_init[index_key].append( float(res_dict["adds_h_init"].cpu()) )
        self._add_s_init[index_key].append( float(res_dict["add_s_h_init"].cpu()) )


        self.log(f'adds_obj'+index_key, self._adds[index_key][-1] , on_step=True, on_epoch=True )
        self.log(f'add_s_obj'+index_key, self._adds[index_key][-1] , on_step=True, on_epoch=True )
        
        self.log(f'avg_adds_obj'+index_key, sum( self._adds[index_key]) / len( self._adds[index_key] ), on_step=True, on_epoch=True )
        self.log(f'avg_add_s_obj'+index_key, sum( self._add_s[index_key]) / len( self._add_s[index_key] ), on_step=True, on_epoch=True )

      self.log(f'count_suc', self.count_suc, on_step=True, on_epoch=True )
      self.log(f'count_failed', self.count_failed, on_step=True, on_epoch=True )
      print( "Count SUC", self.count_suc, " Count FAILED", self.count_failed)
      for k in res_dict.keys():
        # print( "k ", k, " res_dict ", res_dict[k] ," value ", res_dict[k].mean())
        self.log(f'{self._mode}_{k}', res_dict[k].mean(), on_step=True, on_epoch=True, prog_bar=False)

      res_dict, count_invalid = full_pose_estimation( 
        h_gt = h_gt.clone(), 
        h_render = h_render.clone(),
        h_init = h_init.clone(),
        bb = bb, 
        flow_valid = valid.clone(), 
        flow_pred = flow.clone(), 
        flow_gt = flow.clone(),
        idx = idx.clone(),
        K_ren = K_ren,
        K_real = K_real,
        render_d = render_d.clone(),
        model_points = model_points.clone(),
        cfg = self._exp.get("full_pose_estimation", {})
      )
      for k in res_dict.keys():
        self.log(f'{self._mode}_{k}_gt_flow', res_dict[k].mean(), on_step=True, on_epoch=True, prog_bar=True)


    self.log(f'{self._mode}_epe', metrics['epe'], on_step=True, on_epoch=True, prog_bar=True)
    return {'loss': loss, 'pred': flow_predictions, 'target': flow}
  
  def plot(self, flow_gt, flow_pred, img1, img2, valid ):
      if self._logged_images[self._mode] < self._logged_images_max[self._mode]:
        
        for flow, name in zip( [flow_gt, flow_pred[-1]], ["gt", "pred"] ):
          corros = []
          for b in range( img1.shape[0] ):
          
            i1 = img1[b].permute(1,2,0)
            i2 = img2[b].permute(1,2,0)
            va = valid[b]
            fl = flow[b].permute(1,2,0)
            corros.append ( fn(self._visu.plot_corrospondence( fl[:,:,0], fl[:,:,1], 
                va, i1, i2, colorful = True, text=False, res_h =30, res_w=30, 
                min_points=50, jupyter=False, store=False)))

          res = torch.stack( corros ).permute(0, 3, 1, 2)
          img = make_grid( res,nrow=2, padding=5)
          idx = self._logged_images[self._mode] 
          self._visu.plot_image( img= img, store=True, tag=f"{self._mode}_{name}_{idx}_Samples")
          self._logged_images[self._mode] += 1

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

  def test_step(self, batch, batch_idx, dataloader_idx=0):      
    outputs = self.training_step(batch, batch_idx)
    return outputs
  
  def test_step_end( self, outputs ):
    self.log('test_loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, sync_dist_op=None)
  
  def on_test_epoch_start(self):
    self._visu.logger= self.logger
    self._mode = 'test'
    if self._estimate_pose:
      self.trainer.test_dataloaders[0].dataset.estimate_pose = True
      print( "Set dataloader test to return metrices to estimate OBJECT POSE")

      self._adds = {str(k): [] for k in range(22)}
      self._add_s = {str(k): [] for k in range(22)}

      self._adds_init = {str(k): [] for k in range(22)}
      self._add_s_init = {str(k): [] for k in range(22)}


  
  def test_epoch_end(self, outputs):

    properties = {}
    for index_key in self._adds.keys():
      if len( self._adds[index_key] ) > 0:
        properties['MEAN_ADDS__OBJ('+index_key+")_PRED" ] = sum( self._adds[index_key]) / len( self._adds[index_key] )
        properties['MEAN_ADD_S_OBJ('+index_key+")_PRED" ] = sum( self._add_s[index_key]) / len( self._add_s[index_key] )
        
        properties['AUC_ADDS_OBJ('+index_key+")_PRED"] = compute_auc( np.array( self._adds[index_key] ))
        properties['AUC_ADD_S_OBJ('+index_key+")_PRED"] = compute_auc( np.array( self._add_s[index_key] )) 

        properties['2CM_ADDS_OBJ('+index_key+")_PRED"] = compute_percentage( np.array( self._adds[index_key] ))
        properties['2CM_ADD_S_OBJ('+index_key+")_PRED"] = compute_percentage( np.array( self._add_s[index_key] )) 

        properties['MEAN_ADDS__OBJ('+index_key+")_INIT" ] = sum( self._adds_init[index_key]) / len( self._adds_init[index_key] )
        properties['MEAN_ADD_S_OBJ('+index_key+")_INIT" ] = sum( self._add_s_init[index_key]) / len( self._add_s_init[index_key] )
        
        properties['AUC_ADDS_OBJ('+index_key+")_INIT"] = compute_auc( np.array( self._adds_init[index_key] ))
        properties['AUC_ADD_S_OBJ('+index_key+")_INIT"] = compute_auc( np.array( self._add_s_init[index_key] )) 

        properties['2CM_ADDS_OBJ('+index_key+")_INIT"] = compute_percentage( np.array( self._adds_init[index_key] ))
        properties['2CM_ADD_S_OBJ('+index_key+")_INIT"] = compute_percentage( np.array( self._add_s_init[index_key] )) 


    for p in properties.keys():
      self.logger.experiment.set_property( p, properties[p] )
      print( p, ": ", properties[p])

    import pickle
    properties ['self._adds'] = self._adds
    properties ['self._add_s'] = self._add_s
    properties ['self._adds_init'] = self._adds_init
    properties ['self._add_s_init'] = self._add_s_init
    with open(os.path.join(self._exp['name'], 'results.pkl'), 'wb') as handle:
        pickle.dump(properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
    self.logger.experiment.log_artifact( os.path.join(self._exp['name'], 'results.pkl') )

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

      ret = [optimizer], [scheduler]
    else:
      ret = [optimizer]
    return ret
  
  