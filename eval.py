import sys
import os
os.chdir(os.path.join(os.getenv('HOME'), 'RPOSE'))
sys.path.append('core')

import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils.utils import InputPadder

os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'ASL'))
sys.path.append(os.path.join(os.path.join(os.getenv('HOME'), 'ASL') + '/src'))

import yaml
def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def load_yaml(path):
  with open(path) as file:  
    res = yaml.load(file, Loader=yaml.FullLoader) 
  return res

import coloredlogs
coloredlogs.install()
import time
import argparse
from pathlib import Path
import gc

# Frameworks
import torch
from torchvision import transforms
from torchvision import transforms as tf
import numpy as np
import imageio
# Costume Modules

from datasets_asl import get_dataset
import cv2
import time

def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])

DEVICE = 'cuda:1'

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser() 
	parser.add_argument('--eval', type=file_path, default="/home/jonfrey/ASL/cfg/eval/eval.yml",
											help='Yaml containing dataloader config')
	args = parser.parse_args()
	env_cfg_path = os.path.join('cfg/env', os.environ['ENV_WORKSTATION_NAME']+ '.yml')
	env_cfg = load_yaml(env_cfg_path)	
	eval_cfg = load_yaml(args.eval)

	# SETUP MODEL
	di = {
    'model': '/home/jonfrey/RPOSE/models/raft-things.pth',
    'small': False,
    'mixed_precision': False,
    'alternate_corr': False,
	}    
	class DotDict(dict):
			"""dot.notation access to dictionary attributes"""
			__getattr__ = dict.get
			__setattr__ = dict.__setitem__
			__delattr__ = dict.__delitem__
    
	args = DotDict(di)
	os.chdir(os.path.join(os.getenv('HOME'), 'RPOSE'))
	model = torch.nn.DataParallel(RAFT(args))
	model.load_state_dict(torch.load(args.model))
	model = model.module
	model.to(DEVICE)
	model.eval()
	os.chdir(os.path.join(os.getenv('HOME'), 'ASL'))

	# SETUP DATALOADER
	dataset_test = get_dataset(
	**eval_cfg['dataset'],
	env = env_cfg,
	output_trafo = None,
	)
	sub = eval_cfg['dataset']['sub']
	dataloader_test = torch.utils.data.DataLoader(dataset_test,
	shuffle = False,
	num_workers = 0,
	pin_memory = eval_cfg['loader']['pin_memory'],
	batch_size = 1, 
	drop_last = True)

	# CREATE RESULT FOLDER
	base = os.path.join(env_cfg['base'], eval_cfg['name'], eval_cfg['dataset']['name'])

	globale_idx_to_image_path = dataset_test.image_pths



	with torch.no_grad():
		# START EVALUATION
		st = time.time()
		for j, batch in enumerate( dataloader_test ):
			images = batch[0].to(DEVICE)
			target = batch[1].to(DEVICE)
			ori_img = batch[2].to(DEVICE)
			replayed = batch[3].to(DEVICE)
			BS = images.shape[0]
			global_idx = batch[4]

			global_idx_2 = batch[4].clone() + sub
			images_next_frame = images.clone()
			# In average loading 1 Frame per batch
			for b in range(BS):
				flow_valid = torch.ones( (BS), device=DEVICE)
				if (global_idx_2[b] == global_idx).sum() == 1:
					idx = torch.where( (global_idx_2[b] == global_idx) )
					idx = idx[0]
					images_next_frame[b] = images[idx]
				else:
					# Load this frame
					if int( global_idx_2[b]) in dataset_test.global_to_local_idx:
						local_idx = int( dataset_test.global_to_local_idx.index(int( global_idx_2[b])) )
						ba = dataset_test[local_idx]
						print("Loaded", local_idx, ' for b', b)
						images_next_frame[b] = ba[0].to(DEVICE)
					else:
						flow_valid[b] = 0


			images_next_frame *= 255
			images *= 255

			# tra = tf.Resize(( int( images.shape[2]/2) ,int( images.shape[3]/2)))
			# flow_low, flow_up = model(tra( images ), tra( images_next_frame ), iters=20, test_mode=True)
			flow_low, flow_up = model( images, images_next_frame , iters=12, test_mode=True)

			# pred = tra_up(torch.from_numpy(pred)).numpy()
			for b in range(BS):
					if flow_valid[b] == 1:
						img_path = globale_idx_to_image_path[global_idx[b]]
						p = os.path.join(base,
								img_path.split('/')[-3],
								f'flow_sub_{sub}',
								img_path.split('/')[-1][:-4]+'.png')
						Path(p).parent.mkdir(parents=True, exist_ok=True)
						writeFlowKITTI(p , flow_up[b].permute(1,2,0).cpu())
					else:
						img_path = globale_idx_to_image_path[global_idx[b]]
						p = os.path.join(base,
								img_path.split('/')[-3],
								f'flow_sub_{sub}',
								img_path.split('/')[-1][:-4]+'.png')
						Path(p).parent.mkdir(parents=True, exist_ok=True)
						writeFlowKITTI(p , torch.zeros(flow_up[b].shape ).permute(1,2,0).cpu())	
			if j % 10 == 0:
				print(j, "/" , len(dataloader_test), p)
				print("Estimate Total: ", (time.time()-st)/(1+j)*(len(dataloader_test)),'s' )
				print("Estimate Left: ", (time.time()-st)/(1+j)*(len(dataloader_test)-(1+j) ),'s' )
