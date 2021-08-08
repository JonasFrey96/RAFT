
import os
from pathlib import Path
from PIL import Image
import numpy as np

min_visible_points = 200

mode = 'test_tracking'
p = "/media/scratch2/jonfrey/ycb/data"

scene_nrs = list(range(48,59))

candidates = [str(p) for p in Path(p).rglob('*-label.png') if int(str(p).split("/")[-2]) in scene_nrs]
candidates.sort(key=lambda x: int(x.split('/')[-2])*10000000 + int(x.split('/')[-1][:6])    )


def get_camera(desig, K=False, idx=False):
    """
    make this here simpler for cameras
    """
    
    if desig[:8] != 'data_syn' and int(desig[5:9]) >= 60:
        cx_2 = 323.7872
        cy_2 = 279.6921
        fx_2 = 1077.836
        fy_2 = 1078.189
        if K :
            return np.array([[fx_2,0,cx_2],[0,fy_2,cy_2],[0,0,1]])
        elif idx:
            return 1
        else:
            return np.array([cx_2, cy_2, fx_2, fy_2])
            
    else:
        cx_1 = 312.9869
        cy_1 = 241.3109
        fx_1 = 1066.778
        fy_1 = 1067.487
        if K:
            return np.array([[fx_1,0,cx_1],[0,fy_1,cy_1],[0,0,1]])
        elif idx:
            return 0 
        else:
            return np.array([cx_1, cy_1, fx_1, fy_1])
            

base_path_list = []
obj_idx_list = []
camera_idx_list = []
for j, c in enumerate( candidates):
    unique, unique_counts = np.unique( np.array( Image.open( c ) ), return_counts=True)
    if  mode.find('test') == -1:
        mask = unique_counts > min_visible_points
        sel_indices = unique[mask].tolist()
    else:
        sel_indices = unique.tolist()

    for index in sel_indices:
        if index != 0:
            pose = c.find('/ycb/')
            base_path_list.append( c[pose+5:-10] )
            obj_idx_list.append( index )
            camera_idx_list.append( get_camera( desig= base_path_list[-1], idx=True))
    if j % 100 == 0:
        print(j, "/", len(candidates))



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

with open(f'/home/jonfrey/RPOSE/cfg/datasets/ycb/{mode}.pkl', 'wb') as handle:
    pickle.dump( dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

