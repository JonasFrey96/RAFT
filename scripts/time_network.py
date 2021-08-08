import os
import sys 
os.chdir(os.path.join(os.getenv('HOME'), 'RPOSE'))
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/core'))
sys.path.append(os.path.join(os.getcwd() + '/segmentation'))


from src_utils import DotDict
from raft import RAFT
import numpy as np

model = RAFT(args = DotDict( {'small':False}) )

import torch
device = 'cuda:0'
BS,H,W,C = 1,480,640,3
half = True

inp1 = torch.randn(BS, C,H,W, dtype=torch.float).to(device)
inp2 = torch.randn(BS, C,H,W, dtype=torch.float).to(device)
model.to(device)
model.eval()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print( pytorch_total_params )


def time_model( model, inp, repetitions = 10):
	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	timings=np.zeros((repetitions,1))
	#GPU-WARM-UP
	for _ in range(50):
			_ = model( *inp)

	# MEASURE PERFORMANCE
	with torch.no_grad():
			for rep in range(repetitions):
					starter.record()
					_ = model(*inp)
					ender.record()
					# WAIT FOR GPU SYNC
					torch.cuda.synchronize()
					curr_time = starter.elapsed_time(ender)
					timings[rep] = curr_time

	mean_syn = np.sum(timings) / repetitions
	std_syn = np.std(timings)
	print(mean_syn, std_syn, timings.min(), timings.max())
	print("HZ: ", 1/(mean_syn/1000) , " STD in ms : ",(std_syn),  " STD in hz : ",1/(std_syn/1000))

print("\nFlow 24")
time_model( model, (inp1,inp2,24), repetitions = 100)
print("\nFlow 12")
time_model( model, (inp1,inp2,12), repetitions = 100)
print("\nFlow 6")
time_model( model, (inp1,inp2,6), repetitions = 100)
print("\nFlow 2")
time_model( model, (inp1,inp2,2), repetitions = 100)

from models_asl import FastSCNN
model = FastSCNN(num_classes= 2, aux = False, extraction = {"active":False,
			"layer":'learn_to_down'}, input_channels =  6)

model.to(device)
model.eval()
print("\nSEGMENTATION")
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print( pytorch_total_params )
time_model( model, (torch.cat ( [inp1,inp2],dim=1),), repetitions = 1000)
