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

from models_asl import FastSCNN

from ycb.rotations import so3_relative_angle
import shutil

__all__ = ['Network']

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid,  synthetic, gamma=0.8, max_flow=MAX_FLOW):
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
    epe2 = epe.clone()
    epe2 = epe2 * valid
    epe2 = epe2.sum(dim=(1,2)) / valid.sum(dim=(1,2))
    metrics = {}
    
    if synthetic.sum() > 0:
      metrics['epe_render'] = epe2[synthetic].mean().item()
    non_synthetic = (synthetic==False)
    if non_synthetic.sum() > 0:
      metrics['epe_real'] = epe2[non_synthetic].mean().item()

    epe = epe.view(-1)[valid.view(-1)]
    metrics['epe'] = epe.mean().item()
    metrics['1px'] = (epe < 1).float().mean().item()
    metrics['3px'] = (epe < 3).float().mean().item()
    metrics['5px'] = (epe < 5).float().mean().item()
    
    return flow_loss, metrics, epe2


class Network(LightningModule):
  def __init__(self, exp, env):
    super().__init__()
    self._exp = exp
    self._env = env
    self.hparams['lr'] = self._exp['lr']
    
    self.model = RAFT(args = DotDict(self._exp['model']['args']) )
    
    self._mode = 'train'
    self._logged_images = {'train': 0, 'val': 0, 'test': 0}
    
    if "logged_images_max" in self._exp.keys():
      self._logged_images_max = self._exp['logged_images_max']
    else:
      self._logged_images_max = {'train': 2, 'val': 2, 'test': 2}
    
    self._type = torch.float16 if exp['trainer'].get('precision',32) == 16 else torch.float32
    self._visu = Visualizer( os.path.join ( exp['name'], "visu"), num_classes=2 ,store=False)

    if self._exp.get('mode','train') == 'test':
      self._estimate_pose = True
          # SEGMENTATION
      self.seg = FastSCNN(**self._exp['seg']['cfg'])
      self.output_transform_seg = transforms.Compose([
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
      ])
    else:
      self._estimate_pose = False
    
    self._count_real = {'train': 0, 'val': 0, 'test': 0}
    self._count_render = {'train': 0, 'val': 0, 'test': 0}
    
    shutil.rmtree('/home/jonfrey/tmp/ycb', ignore_errors=True)
    
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
    synthetic = batch[4]
    flow_predictions = self(batch = batch)

    loss, metrics, epe_per_object = sequence_loss(flow_predictions, flow, valid, synthetic, self._exp['model']['gamma'])

    if self._estimate_pose:
      
      # PRED FLOW 
      inp = torch.cat ( [self.output_transform_seg(batch[0]/255.0),
      self.output_transform_seg(batch[1]/255.0 ) ],dim=1)
      outputs = self.seg(inp)
      probs = torch.nn.functional.softmax(outputs[0], dim=1)
      pred_valid = torch.argmax( probs, dim = 1)
      acc = (pred_valid == valid).sum() / torch.numel( valid)
      h_gt, h_render, h_init, bb, idx, K_ren, K_real, render_d, model_points, img_real_ori, p = batch[5:]
      
      # ESTIMATE POSE
      res_dict, count_invalid, h_pred__pred_pred, ratios = full_pose_estimation( 
        h_gt = h_gt.clone(), 
        h_render = h_render.clone(),
        h_init = h_init.clone(),
        bb = bb, 
        flow_valid = pred_valid.clone(), 
        flow_pred = flow_predictions[-1].clone(), 
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
      
      self.log(f'acc_mask', 
        acc.item(), 
        on_step=True, on_epoch=True )

      index_key = str( int( idx ))
      self.log(f'inital_trans_error_obj'+index_key, 
        ( torch.norm ( h_gt[0,:3,3] - h_init[0,:3,3])).item(), 
        on_step=True, on_epoch=True )
      self.log(f'inital_rotation_error_obj'+index_key, 
        (so3_relative_angle(h_gt[:,:3,:3].type(torch.float32),h_init[:,:3,:3].type(torch.float32) )/ np.math.pi * 180).item() , 
        on_step=True, on_epoch=True )
      
      self.log(f'inital_trans_error', 
        torch.norm ( h_gt[0,:3,3] - h_init[0,:3,3]).item(), 
        on_step=True, on_epoch=True )
      self.log(f'inital_rotation_error', 
        (so3_relative_angle(h_gt[:,:3,:3].type(torch.float32),h_init[:,:3,:3].type(torch.float32) ) / np.math.pi * 180).item() , 
        on_step=True, on_epoch=True )

      if len( res_dict ) > 0: 
        self.log(f'ransac_inlier_ratio', float(ratios[0]), on_step=False, on_epoch=True )
        # STORE PREDICTIONS
        tmp = os.path.join ( self._exp['name'] , p[0][ p[0].find('ycb'):], str(int( idx[0] ) +1) + '.npy' )
        tmp2 = os.path.join ( "/home/jonfrey/tmp", p[0][ p[0].find('ycb'):], str(int( idx[0] ) +1) + '.npy' )
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        np.save(str(tmp), h_pred__pred_pred[0].cpu().numpy())
        Path(tmp2).parent.mkdir(parents=True, exist_ok=True)
        np.save(str(tmp2), h_pred__pred_pred[0].cpu().numpy())
        
        index_key = str( int( idx ))
        self.log(f'acc_mask_obj' + index_key, 
          acc.item(), 
          on_step=True, on_epoch=True )
        
        self.log(f'adds_init_obj'+index_key, res_dict["adds_h_init"].cpu().item() , on_step=True, on_epoch=True )
        self.log(f'add_s_init_obj'+index_key, res_dict["add_s_h_init"].cpu().item() , on_step=True, on_epoch=True )

        self.log(f'adds_obj'+index_key, res_dict["adds_h_pred"].cpu().item() , on_step=True, on_epoch=True )
        self.log(f'add_s_obj'+index_key, res_dict["add_s_h_pred"].cpu().item() , on_step=True, on_epoch=True )
        self.plot_pose(
          model_points = model_points,
          h_gt = h_gt,
          h_init = h_init,
          h_pred = h_pred__pred_pred,
          pred_valid = pred_valid,
          img_real_zoom = batch[0],
          img_real_ori = img_real_ori,
          K_real = K_real,
          index = batch_idx
        )
      else:
        print( "Count SUC", self.count_suc, " Count FAILED", self.count_failed)        
        # print("Force PLOT since Pose Estimation vailed!")
        # self.plot( batch[2], flow_predictions, batch[0], batch[1], pred_valid, force =True, index = batch_idx)
        # self.plot_seg ( batch[0], batch[1], pred_valid, valid,force = True, index = batch_idx )

      if batch_idx % 50 == 0:
        self.log(f'count_suc', self.count_suc, on_step=True, on_epoch=False )
        self.log(f'count_failed', self.count_failed, on_step=True, on_epoch=False )

      for k in res_dict.keys():
        # print( "k ", k, " res_dict ", res_dict[k] ," value ", res_dict[k].mean())
        self.log(f'{self._mode}_{k}_pred_flow_pred_seg', res_dict[k].mean().item(), on_step=True, on_epoch=False, prog_bar=False)
      
      if False:
        # GT FLOW GT SEG
        res_dict, count_invalid, h_pred__gt_gt = full_pose_estimation( 
          h_gt = h_gt.clone(), 
          h_render = h_render.clone(),
          h_init = h_init.clone(),
          bb = bb, 
          flow_valid = valid.clone(), 
          flow_pred = flow.clone(), 
          idx = idx.clone(),
          K_ren = K_ren,
          K_real = K_real,
          render_d = render_d.clone(),
          model_points = model_points.clone(),
          cfg = self._exp.get("full_pose_estimation", {})
        )
        for k in res_dict.keys():
          self.log(f'{self._mode}_{k}_gt_flow_gt_seg', res_dict[k].mean(), on_step=True, on_epoch=True, prog_bar=True)

        # PRED FLOW GT SEG
        h_gt, h_render, h_init, bb, idx, K_ren, K_real, render_d, model_points, img_real_ori, p = batch[5:]
        res_dict, count_invalid, h_pred__pred_gt = full_pose_estimation( 
          h_gt = h_gt.clone(), 
          h_render = h_render.clone(),
          h_init = h_init.clone(),
          bb = bb, 
          flow_valid = valid.clone(), 
          flow_pred = flow_predictions[-1].clone(), 
          idx = idx.clone(),
          K_ren = K_ren,
          K_real = K_real,
          render_d = render_d.clone(),
          model_points = model_points.clone(),
          cfg = self._exp.get("full_pose_estimation", {})
        )
        for k in res_dict.keys():
          self.log(f'{self._mode}_{k}_pred_flow_gt_seg', res_dict[k].mean(), on_step=True, on_epoch=True, prog_bar=False)
        
    else:
      idx = batch[5]
      
    logging_metrices = ['epe', 'epe_real', 'epe_render']
    for met in logging_metrices:
      if met in metrics:
        self.log(f'{self._mode}_{met}', metrics[met], on_step=True, on_epoch=False, prog_bar=True)
    
    if self._exp.get( 'log',{}).get('individual_obj',{}).get(self._mode, False):
      for i in range(BS):
        obj = str(int(idx[i]))
        self.log(f'{self._mode}_{met}_obj{obj}', epe_per_object[i].float().item(), on_step=True, on_epoch=False, prog_bar=True)
          
    self._count_real[self._mode] += (synthetic ==False).sum()
    self._count_render[self._mode] += (synthetic).sum()
    
    self.log(f'{self._mode}_count_real', self._count_real[self._mode], on_step=False, on_epoch=False, prog_bar=False)
    self.log(f'{self._mode}_count_render', self._count_render[self._mode], on_step=False, on_epoch=False, prog_bar=False)
    
    return {'loss': loss, 'pred': flow_predictions, 'target': flow}


  def plot_pose(self, model_points, h_gt, h_init, h_pred, pred_valid, img_real_zoom, img_real_ori, K_real, index=0 ):
      if self._logged_images[self._mode] < self._logged_images_max[self._mode]:
        b = 0
        img_gt = self._visu.plot_estimated_pose( 
            img = img_real_ori[b].cpu().numpy(), 
            points = model_points[b].cpu(), 
            H = h_gt[b].cpu(),
            K = K_real[b].cpu(), 
            tag = 'Test_gt',
            epoch = index,
            not_log = True,
            store = False)
        img_pred = self._visu.plot_estimated_pose( 
            img = img_real_ori[b].cpu().numpy(), 
            points = model_points[b].cpu(), 
            H = h_pred[b].cpu(),
            K = K_real[b].cpu(), 
            tag = 'Test_pred',
            epoch = index, 
            not_log = True, store= False)

        img_init = self._visu.plot_estimated_pose( 
            img = img_real_ori[b].cpu().numpy(), 
            points = model_points[b].cpu(), 
            H = h_init[b].cpu(),
            K = K_real[b].cpu(), 
            tag = 'Test_init',
            not_log = True, store= False)

        ass = np.concatenate( [img_init, img_pred, img_gt], axis = 1)
        print(ass.shape)
        self._visu.plot_image( img= ass, tag = 'Pose_INIT_PRED_GT', epoch=index, store= False)

  def plot(self, flow_gt, flow_pred, img1, img2, valid ,force = False ):
      if self._logged_images[self._mode] < self._logged_images_max[self._mode] or force:
        
        for flow, name in zip( [flow_gt, flow_pred[-1]], ["gt", "pred"] ):
          corros = []
          for b in range( img1.shape[0] ):
          
            i1 = img1[b].permute(1,2,0)
            i2 = img2[b].permute(1,2,0)
            va = valid[b]
            fl = flow[b].permute(1,2,0)
            corros.append ( fn(self._visu.plot_corrospondence( fl[:,:,0], fl[:,:,1], 
                va, i1, i2, colorful = True, text=False, res_h =30, res_w=30, 
                min_points=50, jupyter=False, not_log=True)))

          res = torch.stack( corros ).permute(0, 3, 1, 2)
          img = make_grid( res,nrow=2, padding=5)
          idx = self._logged_images[self._mode] 
          
          
          nr = self._logged_images[self._mode] + self.trainer.current_epoch * (self._logged_images_max[self._mode] + 1)
          self._visu.plot_image( img= img, tag=f"Flow_{self._mode}_{name}", epoch= nr, store= False )
          self._logged_images[self._mode] += 1

  def plot_seg(self, ori_real, ori_render, pred, target,force = False, idx = None, index= 0):
    if self._logged_images[self._mode] < self._logged_images_max[self._mode] or force:
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
      
      self._visu.plot_detectron( img = grid_ori_real , label = grid_pred[0,:,:] , tag = 'PRED SEG', method="left", store= False)
      self._visu.plot_image( img = grid_ori_render , tag='Segmentation_left_pred__right_render_img', method= 'right', epoch= index, store= False )

      self._visu.plot_detectron( img = grid_ori_real , label = grid_pred[0,:,:] , tag = 'PRED SEG', method="left", store= False)
      self._visu.plot_detectron( img = grid_ori_real , label = grid_target[0,:,:],  tag='Segmentation_left_pred__right_gt',  method="right", epoch= index , store= False)

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
        properties['MEAN_ADDS__OBJ('+index_key+")_PRED_CM" ] = round( sum( self._adds[index_key]) / len( self._adds[index_key] ) *100,2)
        properties['MEAN_ADDS__OBJ('+index_key+")_INIT_CM" ] = round( sum( self._adds_init[index_key]) / len( self._adds_init[index_key] ) *100,2)
    for index_key in self._adds.keys():
      if len( self._adds[index_key] ) > 0:        
        properties['MEAN_ADD_S_OBJ('+index_key+")_PRED_CM" ] = round( sum( self._add_s[index_key]) / len( self._add_s[index_key] ) *100,2)
        properties['MEAN_ADD_S_OBJ('+index_key+")_INIT_CM" ] = round( sum( self._add_s_init[index_key]) / len( self._add_s_init[index_key] ) *100,2)
    for index_key in self._adds.keys():
      if len( self._adds[index_key] ) > 0:
        properties['AUC_ADDS_OBJ('+index_key+")_PRED"] = round( compute_auc( np.array( self._adds[index_key] )),2)
        properties['AUC_ADDS_OBJ('+index_key+")_INIT"] = round( compute_auc( np.array( self._adds_init[index_key] )),2)
    for index_key in self._adds.keys():
      if len( self._adds[index_key] ) > 0:
        properties['AUC_ADD_S_OBJ('+index_key+")_PRED"] = round( compute_auc( np.array( self._add_s[index_key] )) ,2)
        properties['AUC_ADD_S_OBJ('+index_key+")_INIT"] = round( compute_auc( np.array( self._add_s_init[index_key] )) ,2)
    for index_key in self._adds.keys():
      if len( self._adds[index_key] ) > 0:
        properties['2CM_ADDS_OBJ('+index_key+")_PRED"] = round( compute_percentage( np.array( self._adds[index_key] ))  ,2)
        properties['2CM_ADDS_OBJ('+index_key+")_INIT"] = round( compute_percentage( np.array( self._adds_init[index_key] ))  ,2)
    for index_key in self._adds.keys():
      if len( self._adds[index_key] ) > 0:
        properties['2CM_ADD_S_OBJ('+index_key+")_PRED"] = round( compute_percentage( np.array( self._add_s[index_key] ))  ,2)
        properties['2CM_ADD_S_OBJ('+index_key+")_INIT"] = round( compute_percentage( np.array( self._add_s_init[index_key] ))  ,2)
    kk = list( properties.keys())
    kk.sort()
    for p in kk :
      self.logger.experiment.set_property( p, properties[p] )

      if p.find('PRED') != -1:
        print( p.replace("PRED","") , " INIT: " , properties[p.replace('PRED','INIT')], " PRED: ",  properties[p])
      # print( p, ": ", properties[p])

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
  
  