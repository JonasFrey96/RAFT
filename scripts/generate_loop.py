import os
import numpy as np

import time

while True:
	if os.path.exists("/home/jonfrey/tmp/nr.npy"):
		start = np.load( "/home/jonfrey/tmp/nr.npy" )
	else:
		lam = 3500
		start= np.array( [ lam ] )
	
	if start[0] < 100000:
		os.system("python /home/jonfrey/RPOSE/scripts/generate_synthetic_data.py")
		time.sleep(5)
	else:
		break