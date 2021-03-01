import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))

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
import gc

# Frameworks
import torch

# Costume Modules
from utils import file_path, load_yaml

if __name__ == "__main__":
  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='/home/jonfrey/ASL/cfg/exp/scannet/scannet.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                      help='The environment yaml file.')
  parser.add_argument('--mode', default='module', choices=['shell','module'],
                      help='The environment yaml file.')
  
  args = parser.parse_args()
  exp = load_yaml(args.exp)
  env = load_yaml(args.env)
  if exp['max_tasks'] > exp['task_generator']['total_tasks']:
    print('Max Tasks larger then total tasks -> Setting max_tasks to total_tasks')
    exp['max_tasks'] = exp['task_generator']['total_tasks']

  exp = load_yaml(args.exp)
  env = load_yaml(args.env)
  
  
  from task import TaskCreator
  seed_everything(42)
  local_rank = int(os.environ.get('LOCAL_RANK', 0))
  if local_rank != 0 or not init:
    print(init, local_rank)
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])
  
  

  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

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
  
  

  # if local_rank == 0 and env['workstation'] == False:
  #     cm = open(os.path.join(model_path, f'info{local_rank}_{task_nr}.log'), 'w')
  # else:
  #     cm = nullcontext()
  # with cm as f:
  #   if local_rank == 0 and env['workstation'] == False:
  #     cm2 = redirect_stdout(f)
  #   else:
  #     cm2 = nullcontext()
  #   with cm2:
  # # Setup logger for each ddp-task 
  # logging.getLogger("lightning").setLevel(logging.DEBUG)
  # logger = logging.getLogger("lightning")
  # fh = logging.FileHandler( , 'a')
  # logger.addHandler(fh)
      
  # Copy Dataset from Scratch to Nodes SSD

  if env['workstation'] == False:
    # use proxy hack for neptunai !!!
    # move data to ssd
    if exp['move_datasets'][0]['env_var'] != 'none':
      for dataset in exp['move_datasets']:
        scratchdir = os.getenv('TMPDIR')
        
        print( 'TMPDIR directory: ', scratchdir )
        env_var = dataset['env_var']
        tar = os.path.join( env[env_var],f'{env_var}.tar')
        name = (tar.split('/')[-1]).split('.')[0]
          
        if not os.path.exists(os.path.join(scratchdir,dataset['env_var']) ):
          
          try:  
            cmd = f"tar -xvf {tar} -C $TMPDIR >/dev/null 2>&1"
            st =time.time()
            print( f'Start moveing dataset-{env_var}: {cmd}')
            os.system(cmd)
            env[env_var] = str(os.path.join(scratchdir, name))
            new_env_var = env[env_var]
            print( f'Finished moveing dataset-{new_env_var} in {time.time()-st}s')
            
          except:
              rank_zero_warn( 'ENV Var'+ env_var )
              env[env_var] = str(os.path.join(scratchdir, name))
              rank_zero_warn('Copying data failed')
        else:
          env[env_var] = str(os.path.join(scratchdir, name))
    else:
      env['mlhypersim'] = str(os.path.join(env['mlhypersim'], 'mlhypersim'))
      
      
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
  if local_rank == 0:
    for i in range(exp['task_generator']['total_tasks']):
      if i == task_nr:
        m = '/'.join( [a for a in model_path.split('/') if a.find('rank') == -1])
        dic = copy.deepcopy( exp['cb_checkpoint']['cfg'])
        checkpoint_callback = ModelCheckpoint(
          dirpath= m,
          filename= 'task'+str(i)+'-{epoch:02d}--{step:06d}',
          **dic
        )
        
        cb_ls.append( checkpoint_callback )
      
  
  if not exp.get('offline_mode', False):
    # if exp.get('experiment_id',-1) == -1:
      #create new experiment_id and write back
    if  logger_pass is None:
      logger = get_neptune_logger(exp=exp,env=env,
        exp_p =exp_cfg_path, env_p = env_cfg_path, project_name="jonasfrey96/asl")
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
  
  checkpoint_load = exp['checkpoint_load']
  
  if local_rank == 0 and init:
    # write back the exp file with the correct name set to the model_path!
    # other ddp-task dont need to care about timestamps
    # also storeing the path to the latest.ckpt that downstream tasks can restore the model state
    exp['weights_restore_2'] = False
    exp['checkpoint_restore_2'] = True
    exp['checkpoint_load_2'] = os.path.join( model_path,'last.ckpt')
    
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])
    Path(exp_cfg_path).parent.mkdir(parents=True, exist_ok=True) 
    with open(exp_cfg_path, 'w+') as f:
      yaml.dump(exp, f, default_flow_style=False, sort_keys=False)
  
  if not init:
    # restore model state from previous task.
    exp['checkpoint_restore'] = exp['checkpoint_restore_2']
    exp['checkpoint_load'] = exp['checkpoint_load_2']
    exp['weights_restore'] = exp['weights_restore_2']
  
  # Always use advanced profiler
  if exp['trainer'].get('profiler', False):
    exp['trainer']['profiler'] = AdvancedProfiler(output_filename=os.path.join(model_path, 'profile.out'))
  else:
    exp['trainer']['profiler']  = False
  
  # print( exp['trainer'] )
  # print(os.environ.get('GLOBAL_RANK'))
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
    

  if exp['weights_restore'] :
    # it is not strict since the latent replay buffer is not always available
    p = os.path.join( env['base'],exp['checkpoint_load'])
    if os.path.isfile( p ):
      res = model.load_state_dict( torch.load(p,
        map_location=lambda storage, loc: storage)['state_dict'], 
        strict=False)
      print('Restoring weights: ' + str(res))
    else:
      raise Exception('Checkpoint not a file')
  
  main_visu = MainVisualizer( p_visu = os.path.join( model_path, 'main_visu'), 
                            logger=logger, epoch=0, store=True, num_classes=22 )
  
  tc = TaskCreator(**exp['task_generator'],output_size=exp['model']['input_size'])
  print(tc)
  _task_start_training = time.time()
  _task_start_time = time.time()
  
  for idx, out in enumerate(tc):
    if idx == task_nr:
      break 
  
  if True:
  #for idx, out in enumerate(tc):
    task, eval_lists = out
    main_visu.epoch = idx
    # New Logger
    print( f'<<<<<<<<<<<< TASK IDX {idx} TASK NAME : '+task.name+ ' >>>>>>>>>>>>>' )

    model._task_name = task.name
    model._task_count = idx
    dataloader_train, dataloader_buffer= get_dataloader_train(d_train= task.dataset_train_cfg,
                                                                env=env,exp = exp)
    print(str(dataloader_train.dataset))
    print(str(dataloader_buffer.dataset))
    dataloader_list_test = eval_lists_into_dataloaders(eval_lists, env=env, exp=exp)
    print( f'<<<<<<<<<<<< All Datasets are loaded and set up >>>>>>>>>>>>>' )
    #Training the model
    trainer.should_stop = False
    # print("GLOBAL STEP ", model.global_step)
    for d in dataloader_list_test:
      print(str(d.dataset))
    
    
    if idx < exp['start_at_task']:
      # trainer.limit_val_batches = 1.0
      trainer.limit_train_batches = 1
      trainer.max_epochs = 1
      trainer.check_val_every_n_epoch = 1
      train_res = trainer.fit(model = model,
                              train_dataloader= dataloader_train,
                              val_dataloaders= dataloader_list_test)
      
      trainer.max_epochs = exp['trainer']['max_epochs']
      trainer.check_val_every_n_epoch =  exp['trainer']['check_val_every_n_epoch']
      trainer.limit_val_batches = exp['trainer']['limit_val_batches']
      trainer.limit_train_batches = exp['trainer']['limit_train_batches']
    else:
      print('Train', dataloader_train)
      print('Val', dataloader_list_test)
      
      train_res = trainer.fit(model = model,
                              train_dataloader= dataloader_train,
                              val_dataloaders= dataloader_list_test)
    res = trainer.logger_connector.callback_metrics
    res_store = {}
    for k in res.keys():
      try:
        res_store[k] = float( res[k] )
      except:
        pass
    base_path = '/'.join( [a for a in model_path.split('/') if a.find('rank') == -1])
    with open(f"{base_path}/res{task_nr}.pkl", "wb") as f:
      pickle.dump(res_store, f)
    
    print( f'<<<<<<<<<<<< TASK IDX {idx} TASK NAME : '+task.name+ ' Trained >>>>>>>>>>>>>' )

    if exp.get('buffer',{}).get('fill_after_fit', False):
      print( f'<<<<<<<<<<<< Performance Test to Get Buffer >>>>>>>>>>>>>' )
      
      trainer.test(model=model,
                  test_dataloaders= dataloader_buffer)
    
      if local_rank == 0:
        checkpoint_callback.save_checkpoint(trainer, model)
      print( f'<<<<<<<<<<<< Performance Test DONE >>>>>>>>>>>>>' )
    
    number_validation_dataloaders = len( dataloader_list_test ) 
    
    if model._rssb_active:
      # visualize rssb
      bins, valids = model._rssb.get()
      fill_status = (bins != 0).sum(axis=1)
      main_visu.plot_bar( fill_status, x_label='Bin', y_label='Filled', title='Fill Status per Bin', sort=False, reverse=False, tag='Buffer_Fill_Status')
    
    plot_from_pkl(main_visu, base_path, task_nr)
    
    validation_acc_plot(main_visu, logger)
  
  try:
    if close:
      logger.experiment.stop()
  except:
    pass


if __name__ == "__main__":
  def signal_handler(signal, frame):
    print('exiting on CRTL-C')
    sys.exit(0)
  print("CALLED TRAIN TASK AS MAIN")
  # this is needed for leonhard to use interactive session and dont freeze on
  # control-C !!!!
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  parser = argparse.ArgumentParser()    
  parser.add_argument('--exp', type=file_path, default='cfg/exp/exp.yml',
                      help='The main experiment yaml file.')
  parser.add_argument('--env', type=file_path, default='cfg/env/env.yml',
                      help='The environment yaml file.')
  
  args = parser.parse_args()
	exp_cfg_path = args.exp
	env_cfg_path = args.env
  
  local_rank = int(os.environ.get('LOCAL_RANK', 0))

  if local_rank != 0:
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])      
  
  exp = load_yaml(exp_cfg_path)
  env = load_yaml(env_cfg_path)

 # TIMESTAMP + INIT EXP FOLDER
  if local_rank == 0:
    seed_everything(42)
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
  
  
  # COPY DATASET
  if env['workstation'] == False:
    # use proxy hack for neptunai !!!
    # move data to ssd
    if exp['move_datasets'][0]['env_var'] != 'none':
      for dataset in exp['move_datasets']:
        scratchdir = os.getenv('TMPDIR')
        
        print( 'TMPDIR directory: ', scratchdir )
        env_var = dataset['env_var']
        tar = os.path.join( env[env_var],f'{env_var}.tar')
        name = (tar.split('/')[-1]).split('.')[0]
          
        if not os.path.exists(os.path.join(scratchdir,dataset['env_var']) ):
          
          try:  
            cmd = f"tar -xvf {tar} -C $TMPDIR >/dev/null 2>&1"
            st =time.time()
            print( f'Start moveing dataset-{env_var}: {cmd}')
            os.system(cmd)
            env[env_var] = str(os.path.join(scratchdir, name))
            new_env_var = env[env_var]
            print( f'Finished moveing dataset-{new_env_var} in {time.time()-st}s')
            
          except:
              rank_zero_warn( 'ENV Var'+ env_var )
              env[env_var] = str(os.path.join(scratchdir, name))
              rank_zero_warn('Copying data failed')
        else:
          env[env_var] = str(os.path.join(scratchdir, name))
    else:
      env['mlhypersim'] = str(os.path.join(env['mlhypersim'], 'mlhypersim'))
      
      
  if ( exp['trainer'] ).get('gpus', -1):
    nr = torch.cuda.device_count()
    exp['trainer']['gpus'] = nr
    print( f'Set GPU Count for Trainer to {nr}!' )
    

  model = Network(exp=exp, env=env)
  
  # LR MONITOR
  lr_monitor = LearningRateMonitor(
    **exp['lr_monitor']['cfg'])
  if exp['cb_early_stopping']['active']:
    early_stop_callback = EarlyStopping(
    **exp['cb_early_stopping']['cfg']
    )
    cb_ls = [early_stop_callback, lr_monitor]
  else:
    cb_ls = [lr_monitor]

	# CHECKPOINT
	if local_rank == 0:
    for i in range(exp['task_generator']['total_tasks']):
      if i == task_nr:
        m = '/'.join( [a for a in model_path.split('/') if a.find('rank') == -1])
        dic = copy.deepcopy( exp['cb_checkpoint']['cfg'])
        checkpoint_callback = ModelCheckpoint(
          dirpath= m,
          filename= 'task'+str(i)+'-{epoch:02d}--{step:06d}',
          **dic
        )
        cb_ls.append( checkpoint_callback )
	
	# GET LOGGER
  if not exp.get('offline_mode', False):
		logger = get_neptune_logger(exp=exp,env=env,
			exp_p =exp_cfg_path, env_p = env_cfg_path, project_name="jonasfrey96/asl")
		exp['experiment_id'] = logger.experiment.id
		print('created experiment id' +  str( exp['experiment_id']))
  else:
    logger = TensorBoardLogger(
      save_dir=model_path,
      name= 'tensorboard', # Optional,
      default_hp_metric=params, # Optional,
    )
  
  # WRITE BACK EXP FILE FOR OTHER ranks!
  if local_rank == 0 and init:
    # write back the exp file with the correct name set to the model_path!
    # other ddp-task dont need to care about timestamps
    # also storeing the path to the latest.ckpt that downstream tasks can restore the model state
    rm = exp_cfg_path.find('cfg/exp/') + len('cfg/exp/')
    exp_cfg_path = os.path.join( exp_cfg_path[:rm],'tmp/',exp_cfg_path[rm:])
    Path(exp_cfg_path).parent.mkdir(parents=True, exist_ok=True) 
    with open(exp_cfg_path, 'w+') as f:
      yaml.dump(exp, f, default_flow_style=False, sort_keys=False)
  
  # Always use advanced profiler
  if exp['trainer'].get('profiler', False):
    exp['trainer']['profiler'] = AdvancedProfiler(output_filename=os.path.join(model_path, 'profile.out'))
  else:
    exp['trainer']['profiler']  = False
  
  # print( exp['trainer'] )
  # print(os.environ.get('GLOBAL_RANK'))
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
    

  if exp['weights_restore'] :
    # it is not strict since the latent replay buffer is not always available
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
    if close:
      logger.experiment.stop()
  except:
    pass