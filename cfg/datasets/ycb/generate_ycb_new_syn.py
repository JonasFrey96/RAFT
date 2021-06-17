from pathlib import Path
import scipy.io as scio

p = "/home/jonfrey/Datasets/ycb/data_syn_new"
candidates = [str(p) for p in Path(p).rglob('*-label.png')]

base_path_list = []
obj_idx_list = []
camera_idx_list = []
for j, c in enumerate( candidates):
    meta = scio.loadmat( c.replace('-label.png', '-meta.mat') )
    
    index = meta["cls_indexes"][0,0]
    pose = c.find('/ycb/')
    base_path_list.append( c[pose+5:-10] )
    obj_idx_list.append( index )
    camera_idx_list.append( 0 )


names_list = [ '002_master_chef_can',
'003_cracker_box',
'004_sugar_box',
'005_tomato_soup_can',
'006_mustard_bottle',
'007_tuna_fish_can',
'008_pudding_box',
'009_gelatin_box',
'010_potted_meat_can',
'011_banana',
'019_pitcher_base',
'021_bleach_cleanser',
'024_bowl',
'025_mug',
'035_power_drill',
'036_wood_block',
'037_scissors',
'040_large_marker',
'051_large_clamp',
'052_extra_large_clamp',
'061_foam_brick' ]
names_idx = { n: idx+1 for (idx, n) in enumerate(names_list) }
idx_names = { n: idx+1 for (idx, n) in enumerate(names_list) }

import pickle
dic = {
'names_idx': names_idx,
'idx_names': idx_names,
'base_path_list': base_path_list,
'obj_idx_list': obj_idx_list,
'camera_idx_list': camera_idx_list}

with open(f'/home/jonfrey/RPOSE/cfg/datasets/ycb/data_syn_new.pkl', 'wb') as handle:
    pickle.dump( dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

