import pickle
import numpy as np
mode = "train_new_syn"

with open(f'cfg/datasets/ycb/{mode}.pkl', 'rb') as handle:
	mappings = pickle.load(handle)
	names_idx = mappings['names_idx']
	idx_names = mappings['idx_names']
	base_path_list = mappings['base_path_list']
	obj_idx_list = mappings['obj_idx_list']
	camera_idx_list = mappings['camera_idx_list']
	

long_ls = []
short_ls = []
for p in base_path_list:
	if not (p.split('/')[-2] in short_ls):
		short_ls.append( p.split('/')[-2] )
		long_ls.append( p )

print(long_ls)
print("Scenes: ", len(long_ls), " Total:", len(base_path_list))

un, co = np.unique( np.array( obj_idx_list ), return_counts=True)
print("Unqiue:", un , " Counts:", co)
# import pdb; pdb.set_trace()
