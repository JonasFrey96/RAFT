import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'RPOSE'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/core'))
sys.path.append(os.path.join(os.getcwd() + '/segmentation'))

import coloredlogs
coloredlogs.install()
from collections import OrderedDict
import time
import shutil
import datetime
import argparse
import signal
import yaml
import logging
from pathlib import Path
import copy

import torch
from src_utils import file_path, load_yaml
import datasets
from lightning import Inferencer
from pose_estimation import full_pose_estimation, compute_auc, compute_percentage
import numpy as np

from enum import Enum
from ycb.rotations import so3_relative_angle
from scipy.stats import special_ortho_group

def expand_to_batch(batch, device):
  ret = []
  for b in batch:
    if torch.is_tensor(b):
      ret.append( b[None].cuda() )
    elif type(b) is tuple:
      new = []
      for el in b:
        new.append( el[None].cuda() )
      ret.append( tuple(new) )
    else:
      ret.append( b ) 
  
  #return not mutable
  return tuple(ret)

class Mode(Enum):
  TRACKING = 1
  REFINEMENT = 2
  MUTIPLE_INIT_POSES = 3

def rel_h (h1,h2):
  return so3_relative_angle(torch.tensor( h1 ) [:3,:3][None], torch.tensor( h2 ) [:3,:3][None])
  
def add_noise(h, nt = 0.01, nr= 30):
  h_noise =np.eye(4)
  while  True:
    x = special_ortho_group.rvs(3).astype(np.float32)
    if abs( float( rel_h(h[:3,:3], x)/(2* float( np.math.pi) )* 360) ) < nr:
      break
  h_noise[:3,:3] = x
  h_noise[:3,3] = np.random.normal(loc=h[:3,3], scale=nt)
  return h_noise

# Implements 
class Evaluator():
  def __init__(self, exp, env):
    super().__init__()
    self._exp = exp
    self._env = env
    self._val = exp.get('val', {})
    self._inferencer = Inferencer(exp, env)
    self.device= 'cuda'
    self._inferencer.to(self.device)
    self.iterations = 4
    self.mode = Mode.MUTIPLE_INIT_POSES  #MUTIPLE_INIT_POSES 
  
  @torch.no_grad()
  def evaluate_full_dataset(self, test_dataloader):
    ycb = test_dataloader.dataset
    ycb.deterministic_random_shuffel()
    print("_base_path_list", ycb._base_path_list)
    ycb.estimate_pose = True 
    ycb.err = True
    ycb.valid_flow_minimum = 0 
    object_to_eval_list = ycb._base_path_list

    elements = len( test_dataloader.dataset._base_path_list )
    adds = np.zeros( (elements,self.iterations) )
    adds[:,:] = np.inf
    add_s = np.zeros( (elements,self.iterations) )
    add_s[:,:] = np.inf
    idx_arr = np.zeros( (elements) )

    init_adds_arr = np.zeros( (elements))
    init_add_s_arr = np.zeros( (elements))
    init_adds_arr[:] = np.inf
    init_add_s_arr[:] = np.inf

    ratios_arr = np.zeros( (elements,self.iterations) )

    valid_corrospondences = np.zeros( (elements,self.iterations) )

    current_pose = torch.eye(4)

    # Iterate over full test dataset list
    for i in range( elements ):
      if i == 200:
        b = f"/home/jonfrey/RPOSE/notebooks/{self.mode}_data.pkl" 
        dic = {
          'add_s' : add_s,
          'adds': adds,
          'idx_arr': idx_arr,
          'ratios_arr': ratios_arr,
          'valid_corrospondences': valid_corrospondences,
        }
        import pickle
        with open(b, 'wb') as handle:
          pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        break

      print(f"Inferenced {i}/{elements}")
      valid_element = True
      # Apply network mutiple iterations
      for k in range ( self.iterations ):
        if k == 0:
          batch = ycb.getElement( i )
          h_store = batch[7].detach().cpu().numpy() # h_init
        else:
          # SET INIT POSE
          if self.mode == Mode.REFINEMENT and valid.sum() != 0:
            # Continue to refine pose
            current_pose = h_pred__pred_pred.detach().cpu().numpy()[0]
          if self.mode == Mode.MUTIPLE_INIT_POSES:
            # we uniformly sample around intial pose
            current_pose = add_noise(h_store, nt = 0.03, nr= 5)
            
          batch = ycb.getElement( i, h_real_est = current_pose )
        
        if batch[0] is None and k == 0:
          print ("Cant start given PoseCNN fails!")
          add_s[i,:] = np.inf
          add[i,:] = np.inf
          idx_arr[i] = int( batch[1] )
          valid_element = False
          break

        if batch[0] is None and k != 0:
          continue
          print("Failed to load data for given pose!")
        
        batch = expand_to_batch(batch, self.device)
        flow_predictions, pred_valid = self._inferencer( batch ) # 200ms
        valid_corrospondences[i,k] = int( pred_valid.sum() )
        gt_valid = batch[3]
        
        h_gt, h_render, h_init, bb, idx, K_ren, K_real, render_d, model_points, img_real_ori, p = batch[5:]
        res_dict, count_invalid, h_pred__pred_pred, ratios, valid = full_pose_estimation( 
          h_gt = h_gt.clone(), 
          h_render = h_render.clone(),
          h_init = h_init.clone(),
          bb = bb, 
          flow_valid = pred_valid.clone(), # TODO: Jonas Frey remove and check why segmentation is so bad
          flow_pred = flow_predictions[-1].clone(), 
          idx = idx.clone(),
          K_ren = K_ren,
          K_real = K_real,
          render_d = render_d.clone(),
          model_points = model_points.clone(),
          cfg = self._exp.get("full_pose_estimation", {})) # 150ms
        if k == 0:
          init_adds_arr[i] = res_dict['adds_h_init']
          init_add_s_arr[i] = res_dict['add_s_h_init']

        if valid.sum() != 0:
          # got vaild prediction 
          adds[i,k] = res_dict['adds_h_pred']
          add_s[i,k] = res_dict['add_s_h_pred']
        else:
          print("Invalid")

        ratios_arr[i,k] = ratios[0]
        
      idx_arr[i] = int( idx[0] )
      # stop because initalizing with PoseCNN failed
      if not valid_element:
        break
      
      if i % 5 == 0 and i != 0:
        print("progress report, ", i)
        
        add_s_finite = np.isfinite( add_s[:i] )
        sm = add_s_finite.sum(axis=1)-1
        sm[sm<0] = 0
        sel = np.eye(self.iterations)[sm] == 1

        print( f"final after {self.iterations}th-iteration: ", compute_auc( add_s[:i][sel]  ) )
        print( "Mean 1th-iteration: ", compute_auc(add_s[:i,0]) )
        print( "AUC best over all iterations: ", compute_auc( np.min( add_s[:i,:], axis = 1) ) )
        tar = np.argmax( ratios_arr[:i], axis = 1 )
        sel = np.zeros_like( ratios_arr[:i] )
        for _j, _i in enumerate( tar.tolist() ):
          sel[_j,_i] = 1
        sel = sel == 1
        print( "Best RANSAC ratios: ", compute_auc( add_s[:i][sel] )  )

        sel2 = np.argmin( valid_corrospondences[:i] ,axis = 1)
        sel2 = np.eye(valid_corrospondences.shape[1])[sel2] == 1
        print( "AUC best valids: ", compute_auc( add_s[:i][sel2] ) )
        print( "INIT ADDS PoseCNN: ", compute_auc( init_add_s_arr[:i] )  )



if __name__ == '__main__':
  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='cfg/exp/exp.yml',
                      help='The main experiment yaml file.')

  args = parser.parse_args()
  exp_cfg_path = args.exp
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

  if exp.get('timestamp',True):
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    model_path = os.path.join(env['base'], exp['name'])
    p = model_path.split('/')
    model_path = os.path.join('/',*p[:-1] ,str(timestamp)+'_'+ p[-1] )
  else:
    model_path = os.path.join(env['base'], exp['name'])
    shutil.rmtree(model_path,ignore_errors=True)
  
  # Create the directory
  Path(model_path).mkdir(parents=True, exist_ok=True)
  
  # Only copy config files for the main ddp-task  
  exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
  env_cfg_fn = os.path.split(env_cfg_path)[-1]
  print(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}')
  shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
  shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')
  exp['name'] = model_path


  inference_manager = Evaluator(exp=exp, env=env)
  
  # LOAD WEIGHTS
  p = os.path.join( env['base'],exp['checkpoint_load'])
  if os.path.isfile( p ):
    res = torch.load( p )
    out = inference_manager._inferencer.load_state_dict( res['state_dict'], 
            strict=False)
    print( "Restore weights from ckpts", out)

  p = os.path.join( env['base'],exp['checkpoint_load_seg'])
  if os.path.isfile( p ):
    res = torch.load( p )
    new_statedict = {}
    for (k,v) in res['state_dict'].items():
      new_statedict[k.replace('model','seg') ] = v
    out = inference_manager._inferencer.load_state_dict( new_statedict, 
            strict=False)
    print( "Restore_seg weights from ckpts", out)

  # PERFORME EVALUATION
  test_dataloader = datasets.fetch_dataloader( exp['test_dataset'], env )
  inference_manager.evaluate_full_dataset(test_dataloader)