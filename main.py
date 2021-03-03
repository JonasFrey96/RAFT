import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'RPOSE'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/core'))

import coloredlogs
coloredlogs.install()

import time
import shutil
import datetime
import argparse
import signal
import yaml
import logging
from pathlib import Path

# Frameworks
import torch

# Costume Modules
from utils import file_path, load_yaml
from lightning import Network

if __name__ == "__main__":
  def signal_handler(signal, frame):
    print('exiting on CRTL-C')
    sys.exit(0)
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  seed_everything(42)

  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='/home/jonfrey/ASL/cfg/exp/scannet/scannet.yml',
                      help='The main experiment yaml file.')

  args = parser.parse_args()
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  env = load_yaml(env_cfg_path)


  
  local_rank = int(os.environ.get('LOCAL_RANK', 0))

  exp_cfg_path = args.exp    
  if local_rank != 0:
    print(init, local_rank)
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])
  exp = load_yaml(exp_cfg_path)

  if local_rank == 0 and init:
    # Set in name the correct model path
    if exp.get('timestamp',True):
      timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
      
      model_path = os.path.join(env['base'], exp['name'])
      p = model_path.split('/')
      model_path = os.path.join('/',*p[:-1] ,str(timestamp)+'_'+ p[-1] )
    else:
      model_path = os.path.join(env['base'], exp['name'])
      try:
        shutil.rmtree(model_path)
      except:
        pass
    # Create the directory
    if not os.path.exists(model_path):
      try:
        os.makedirs(model_path)
      except:
        print("Failed generating network run folder")
    else:
      print("Network run folder already exits")
    
    # Only copy config files for the main ddp-task  
    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]
    print(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}')
    shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
    shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')
    exp['name'] = model_path
  else:
    # the correct model path has already been written to the yaml file.
    model_path = os.path.join( exp['name'], f'rank_{local_rank}_{task_nr}')
    # Create the directory
    if not os.path.exists(model_path):
      try:
        os.makedirs(model_path)
      except:
        pass
  
  # SET NUMBER GPUS
  if ( exp['trainer'] ).get('gpus', -1):
    nr = torch.cuda.device_count()
    exp['trainer']['gpus'] = nr
    print( f'Set GPU Count for Trainer to {nr}!' )
    
  
  model = Network(exp=exp, env=env)
  
  lr_monitor = LearningRateMonitor(
    **exp['lr_monitor']['cfg'])

  if exp['cb_early_stopping']['active']:
    early_stop_callback = EarlyStopping(
    **exp['cb_early_stopping']['cfg']
    )
    cb_ls = [early_stop_callback, lr_monitor]
  else:
    cb_ls = [lr_monitor]
  
  tses = TaskSpecificEarlyStopping(
    nr_tasks=exp['task_generator']['total_tasks'] , 
    **exp['task_specific_early_stopping']
  )
  cb_ls.append(tses)
  if exp['cb_checkpoint']['active']:
    m = '/'.join( [a for a in model_path.split('/') if a.find('rank') == -1])
    dic = copy.deepcopy( )
    checkpoint_callback = ModelCheckpoint(
      dirpath= m,
      filename= 'Checkpoint-{epoch:02d}--{step:06d}',
      **exp['cb_checkpoint']['cfg']
    )
    cb_ls.append( checkpoint_callback )
  
  
  if not exp.get('offline_mode', False):
    if  logger_pass is None:
      logger = get_neptune_logger(exp=exp,env=env,
        exp_p =exp_cfg_path, env_p = env_cfg_path, project_name="jonasfrey96/"+exp['neptune_project_name'] )
      exp['experiment_id'] = logger.experiment.id
      print('created experiment id' +  str( exp['experiment_id']))
    else:
      logger = logger_pass
    print('Neptune Experiment ID: '+ str( logger.experiment.id)+" TASK NR "+str( task_nr ) )
  else:
    logger = TensorBoardLogger(
      save_dir=model_path,
      name= 'tensorboard', # Optional,
      default_hp_metric=params, # Optional,
    )
  
  # WRITE BACK NEW CONF FOR OTHER DDPs
  if local_rank == 0:
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])
    Path(exp_cfg_path).parent.mkdir(parents=True, exist_ok=True) 
    with open(exp_cfg_path, 'w+') as f:
      yaml.dump(exp, f, default_flow_style=False, sort_keys=False)
  
  # PROFILER
  if exp['trainer'].get('profiler', False):
    exp['trainer']['profiler'] = AdvancedProfiler(output_filename=os.path.join(model_path, 'profile.out'))
  else:
    exp['trainer']['profiler']  = False
  
  # TRAINER
  if exp.get('checkpoint_restore', False):
    p = os.path.join( env['base'], exp['checkpoint_load'])
    trainer = Trainer( **exp['trainer'],
      default_root_dir = model_path,
      callbacks=cb_ls, 
      resume_from_checkpoint = p,
      logger=logger)
  else:
    trainer = Trainer(**exp['trainer'],
      default_root_dir=model_path,
      callbacks=cb_ls,
      logger=logger)   
    
  # RESTORE WEIGHTS 
  if exp['weights_restore'] :
    p = os.path.join( env['base'],exp['checkpoint_load'])
    if os.path.isfile( p ):
      res = model.load_state_dict( torch.load(p,
        map_location=lambda storage, loc: storage)['state_dict'], 
        strict=False)
      print('Restoring weights: ' + str(res))
    else:
      raise Exception('Checkpoint not a file')
  
  
  train_res = trainer.fit(model = model,
                          train_dataloader= dataloader_train,
                          val_dataloaders= dataloader_list_test)
  
  try:
      logger.experiment.stop()
  except:
    pass