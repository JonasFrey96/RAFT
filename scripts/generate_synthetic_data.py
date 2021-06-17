import pickle
import numpy as np 
import scipy.io as scio
import os 
import numpy as np

import os
from scipy.spatial.transform import Rotation as R
import random
from math import radians
import time
from pathlib import Path
import scipy
import imageio

# RENDERING
from trimesh import collision
from pyrender.constants import RenderFlags
import trimesh
import pyrender


os.environ["PYOPENGL_PLATFORM"] = "egl"

with open( "/home/jonfrey/RPOSE/cfg/datasets/ycb/test.pkl", "rb") as f:
    res = pickle.load(f)
np.unique( res['base_path_list'] ).shape, len( res['base_path_list'])

paths = np.unique( res['base_path_list'] ).tolist()
base = "/home/jonfrey/Datasets/ycb"

data = []
for j, pa in enumerate( paths ):

    pa = os.path.join( base, pa+'-meta.mat' )
    meta = scio.loadmat( pa )
    
    for k,i in enumerate( meta['cls_indexes'][:,0].tolist()):
        data.append( { 'pose': meta['poses'][:,:,k],
                       'index': i, 
                       'scene_indexes': meta['cls_indexes'][:,0],
                        'path': pa,
                     'intrinsic_matrix': meta['intrinsic_matrix']} )
        
poses = np.array( [d['pose'] for d in data])
indexes = np.array( [d['index'] for d in data])
scene_indexes = np.array( [d['scene_indexes'] for d in data] )


# COMPUTE STATISTICS
stats = []
objects = 21
for i in range(1, objects+1):
    try:
        mask = indexes == i
        me = poses[mask][:,:,3].mean(axis=0)
        std = poses[mask][:,:,3].std(axis=0)

        mi_val = poses[mask][:,:,3].min(axis=0)
        ma_val = poses[mask][:,:,3].max(axis=0)


        count_correlated = np.zeros(( objects ))
        for j in scene_indexes[mask]:
            for n in j:
                count_correlated[n-1] += 1

        # prior equally distributed
        count_correlated += int( count_correlated.sum() /objects )
        count_correlated /= count_correlated.sum() 
        stat = { 'indexes': i-1,
                  'count_correlated': count_correlated,
                  'mean': me,
                'std': std,
                'min_val': mi_val,
               'max_val': ma_val}
        stats.append(stat)
    except:
        pass
      
# LOAD DATA FOR RENDERING
objs_scaled = [ trimesh.load(f'{base}/models/{s}/scaled.obj') 
        for s in os.listdir( "/home/jonfrey/Datasets/ycb/models" ) ]
K = scio.loadmat("/home/jonfrey/Datasets/ycb/data_syn/000001-meta.mat")["intrinsic_matrix"]

objs = [ trimesh.load(f'{base}/models/{s}/textured.obj') 
        for s in os.listdir( "/home/jonfrey/Datasets/ycb/models" ) ]

camera = pyrender.camera.IntrinsicsCamera( K[0,0], K[1,1], K[0,2], K[1,2] )
camera_pose = np.eye(4)
camera_pose[:3,:3 ] = R.from_euler('xyz', [0,180,180], degrees=True).as_matrix()

# RENDERING HELPER AND METHODS

def get_random_light():
    inten = np.random.uniform( 10, 25 , (1,) ) #5 + random.random() * 10
    color = np.random.uniform( 0.7, 1 , (3,) )
    
    if random.random() > 0.3:
        light = pyrender.DirectionalLight(color= color, intensity= inten)
    else:
        light = pyrender.SpotLight(color=color, intensity=inten,
                           innerConeAngle=radians(30+(random.random()*29)),
                           outerConeAngle=radians(60+(random.random()*29)))
    return light

def get_obj_pose(index):
    pose = np.eye(4)
    pose[:3,3] = np.random.uniform( stats[index]['min_val'],stats[index]['max_val'], (3,) )
    pose[:3,:3] = R.random().as_matrix()
    return pose

def get_neighbour_pose(pose):
    pose = np.copy(pose)
    pose[:3,:3] = R.random().as_matrix()
    pose[:3,3] = pose[:3,3] + np.random.uniform( [-0.1,-0.1,-0.3] , [0.1,0.1,0.1], (3,) )
    
    return pose
    
def render():
    try: 
        index = np.random.randint( 0, len(objs) )
        obj = objs[index]
        H,W = 480,640

        # Sample other object according to correlation in dataset
        fac = stats[index]['count_correlated']
        fac[index] = 0
        index2 = np.argmax( np.random.rand(21) * fac )

        # Set metallic randomly
        side_obj = objs[index2] 
        mesh2 = pyrender.Mesh.from_trimesh(side_obj )
        mesh2.primitives[0].material.metallicFactor = random.random() * 0.3
        mesh = pyrender.Mesh.from_trimesh(obj )
        mesh.primitives[0].material.metallicFactor = random.random() * 0.3

        scene = pyrender.Scene()

        # Get random pose uniformly in dataset boundary cube
        obj_pose = get_obj_pose( index )

        # Uniformly sample collision free pose for other object

        cm = collision.CollisionManager()
        cm.add_object(name="obj", mesh=objs_scaled[index], transform=obj_pose)
        while True:
            obj_pose_two = get_neighbour_pose( obj_pose )
            is_collision = cm.in_collision_single(mesh=objs_scaled[index2], transform=obj_pose_two)
            if not is_collision:
                break

        # Add objects to the scene
        st = time.time()
        n_mesh = pyrender.Node(mesh=mesh, matrix= obj_pose  )
        scene.add_node(n_mesh)

        n_mesh2 = pyrender.Node(mesh=mesh2, matrix= obj_pose_two  )
        scene.add_node(n_mesh2)

        # Add camera 
        scene.add(camera, pose=camera_pose)

        # Add random light 
        light_pose = np.copy( camera_pose )
        light_pose[2,3] -= 1
        light_pose[:3,3] += np.random.uniform( 0.3, 0.3 , (3,) )
        scene.add(get_random_light(), pose=light_pose)

        # get mask
        renderer = pyrender.OffscreenRenderer(640, 480)

        flags = RenderFlags.DEPTH_ONLY
        full_depth = renderer.render(scene, flags=flags)
        for mn in scene.mesh_nodes:
            mn.mesh.is_visible = False

        segimg = np.zeros((H, W), dtype=np.uint8)
        for ind, node in zip( [index, index2], [n_mesh, n_mesh2 ] ):
            node.mesh.is_visible = True
            depth = renderer.render(scene, flags=flags)
            mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0
            )
            segimg[mask] = ind + 1
            node.mesh.is_visible = False
        
        if (segimg == index+1).sum() < 100:
          return (False, )
        
        # Show all meshes again
        for mn in scene.mesh_nodes:
            mn.mesh.is_visible = True     

        color, depth = renderer.render(scene)
        ind = np.zeros( (1,1), dtype=np.float32 )
        ind[0,0] = index + 1 
        meta = {
        "cls_indexes": ind, # NR,1
        "factor_depth": np.array([[10000]], dtype=np.float64),
        'poses': obj_pose[:3,:,None].astype(np.float32) ,
        'intrinsic_matrix': np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
            [0.000000e+00, 1.067487e+03, 2.413109e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]]) #3,4,NR
        }

        depth = np.uint16( depth * meta["factor_depth"][0,0] )
        co = np.full(  (H,W,4),255, dtype=np.uint8)
        co[:,:,:3] = color
        del renderer
        time.sleep(0.1)
        
        
        return co, depth, meta, segimg
    except Exception as e:
        print("Failed")
        return (False,)

dir_out = "/home/jonfrey/Datasets/ycb/data_syn_new"

def do(i):
  np.random.seed(i)
  random.seed(i)
  while 1:
    tup = render()
    if len(tup) != 1: break
  color, depth, meta, label = tup

  p_color = os.path.join( dir_out, f"{i:06d}-color.png" )
  p_depth = os.path.join( dir_out, f"{i:06d}-depth.png" )
  p_label = os.path.join( dir_out, f"{i:06d}-label.png" )
  p_meta = os.path.join( dir_out, f"{i:06d}-meta.mat" )

  Path(p_color).parent.mkdir(exist_ok=True, parents= True)
  imageio.imwrite( p_color, color)
  imageio.imwrite( p_depth, depth)
  imageio.imwrite( p_label, label)
  scipy.io.savemat( p_meta, meta)

start_time = time.time()
la = 100000
if os.path.exists("/home/jonfrey/tmp/nr.npy"):
  start = np.load( "/home/jonfrey/tmp/nr.npy" )
  lam = start[0]
else:
  lam = 3500
  start= np.array( [ lam ] )

for j,i in enumerate( range(start[0], la)):
  st = time.time()
  do(i)
  start[0] = i
  np.save( "/home/jonfrey/tmp/nr.npy", start)
  if j > 0:
    print( f"{i} Time gone: ", (time.time()-st), "Time left: ", (time.time()-start_time)/(i-lam)*(la-i) )