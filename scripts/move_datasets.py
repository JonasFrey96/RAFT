import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'RPOSE'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/core'))
from src_utils import file_path, load_yaml
from pathlib import Path
import os, sys
import argparse
import time
def move_datasets(datasets):
  env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
  env = load_yaml(env_cfg_path)

  # COPY DATASET
  if env['workstation'] == False:
    # use proxy hack for neptunai !!!
    # move data to ssd
    for dataset in datasets:
      st =time.time()
      scratchdir = os.getenv('TMPDIR')
      print( 'TMPDIR directory: ', scratchdir )
      
      tar = os.path.join( env[dataset],f'{dataset}.tar')
      name = (tar.split('/')[-1]).split('.')[0]
        
      if not os.path.exists(os.path.join(scratchdir, dataset ) ):
        
        try:  
          cmd = f"tar -xvf {tar} -C $TMPDIR >/dev/null 2>&1"
          
          print( f'Start moveing dataset-{dataset}: {cmd}')
          os.system(cmd)
          
        except:
          print("Failed moveing dataset")
      else:
        print("Path for dataset already exists in TMPDIR")
      
      env[dataset] = str(os.path.join(scratchdir, name))
      new_env_var = env[dataset]
      
      print( f'Finished moveing dataset-{new_env_var} in {time.time()-st}s')
  return env 
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--datasets", nargs="+", default=["ycb_small", "kitti"])
  args = parser.parse_args()
  move_datasets( args.datasets )