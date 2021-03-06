if __name__ == "__main__":
  import os
  import sys
  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + '/src'))
  sys.path.append(os.path.join(os.getcwd() + '/lib'))
  
import time
import random
import copy
import math
import logging
import os
import sys
import pickle
import glob
from pathlib import Path
from PIL import Image

# Frameworks
import numpy as np
import cv2
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R
import scipy.misc
import scipy.io as scio
import torchvision.transforms as transforms
import torch
import torchvision

# For flow calculation
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scipy.interpolate import griddata
import scipy.ndimage as nd

# From costume modules
from ycb_helper import re_quat
from ycb_helper import ViewpointManager
from ycb_helper import get_bb_from_depth, get_bb_real_target
from ycb_helper import Augmentation
from rotations import *







class YCB(torch.data.Dataset):
  def __init__(self, cfg_d, cfg_env):
    super(YCB, self).__init__(cfg_d, cfg_env)
    self._cfg_d = cfg_d
    self._cfg_env = cfg_env
    self._p_ycb = cfg_env['p_ycb']
    
    self._pcd_cad_dict, self._name_to_idx, self._name_to_idx_full = self._get_pcd_cad_models()
    self._batch_list = self._get_batch_list()
    
    self._h = 480
    self._w = 640
    
    self.output_h = cfg_d['output_h'] 
    self.output_w = cfg_d['output_w']
    
    self._aug = Augmentation(add_depth = cfg_d.get('add_depth',False),
                 output_size=(self.output_h, self.output_w), 
                 input_size=(self.output_h, self.output_w))
    
    self._num_pt = cfg_d.get('num_points', 1000)
    self._trancolor_background = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)


    self._min_visible_mask_size = 50

    self._vm = ViewpointManager(
      store=cfg_env['p_ycb'] + '/viewpoints_renderings',
      name_to_idx=self._name_to_idx,
      nr_of_images_per_object=2500,
      device='cpu',
      load_images=False)

    self.K_ren = self.get_camera('data_syn/0019', K=True)
        
    self._load_background()
    self._load_flow()
    self.err = False
    
  def load_lists(self):
    self.path
    self.obj_idx
    self.camera_idx

    self.images
    self.labels
    self.depth

  def __getitem__(self, index):
    return getElement(index, h_real_est=None)

  def getElement(self, index, h_real_est=None):
    """
    desig : sequence/idx
    two problems we face. What is if an object is not visible at all -> meta['obj'] = None
    obj_idx is elemnt 1-21 !!!
    """
    p = self.path[index]
    synthetic = p.find('syn') != -1
    img = Image.open(p+"-color.png")
    depth = np.array( Image.open( p+"-depth.png") )
    label = np.array( Image.open( p+"-label.png") )
    meta = scio.loadmat( p+"-meta.mat")
    obj_idx = self.obj_idx[index]
    K = self.get_camera( self.camera_idx[index] )

    obj = meta['cls_indexes'].flatten().astype(np.int32)
    obj_idx_in_list = int(np.argwhere(obj == obj_idx))
    
    h_gt = np.eye(4)
    h_gt[:3,:4] =  meta['poses'][:, :, obj_idx_in_list]   

    if synthetic:      
      img_arr = np.array( img )[:,:,:3] 
      background_img = self._get_background_image()  
      mask = label == 0
      img_arr[mask] = background_img[mask]
    else:
      img_arr = np.array(img)[:,:,:3]
   
    # if self._cfg_d['output_cfg'].get('add_random_image_as_noise',False):
    #   noise_img = self._get_background_image()
    #   img_arr = img_arr +  ( noise_img - img_arr.mean((0,1))) * 0.025
    
    dellist = [j for j in range(0, len(self._pcd_cad_dict[obj_idx]))]
    dellist = random.sample(dellist, len(
      self._pcd_cad_dict[obj_idx]) - self._num_pt_mesh_large)
    model_points = np.delete(self._pcd_cad_dict[obj_idx], dellist, axis=0).astype(np.float32)
    cam_flag = self.get_camera(desig,K=False,idx=True)
    
    # get rendered data
    res_get_render = self.get_rendered_data( img_arr, depth, label, model_points, int(obj_idx), K_cam, cam_flag, h_gt, h_real_est)
    if res_get_render is False:
      if self.err:
        print("Violation in get render data")
      return False
    
    idx = torch.LongTensor([int(obj_idx) - 1])
    # augment data
    data, uv, flow_mask, gt_label_cropped, non_norm_real_img, non_norm_render_img = \
      self._aug.apply( idx = idx, 
              u_map = res_get_render[5], 
              v_map = res_get_render[6], 
              flow_mask = res_get_render[7],
              gt_label_cropped = res_get_render[4],
              real_img = res_get_render[0], 
              render_img = res_get_render[1],
              real_d = res_get_render[2], 
              render_d = res_get_render[3] 
              )
    output = (
      unique_desig, 
      idx, 
      data, 
      uv, 
      flow_mask,
      gt_label_cropped, 
      non_norm_real_img, 
      non_norm_render_img,
      res_get_render[3], # render_d
      res_get_render[8], # bb
      res_get_render[9], # h_render
      res_get_render[11], # h_gt
      res_get_render[10], # h_init 
      res_get_render[12], # K_real
      torch.from_numpy(model_points), # model_points
    )
    return output

  def get_rendered_data(self, img, depth_real, label, model_points, obj_idx, K_real, cam_flag, h_gt, h_real_est=None):
    """Get Rendered Data
    Args:
      img ([np.array numpy.uint8]): H,W,3
      depth_real ([np.array numpy.int32]): H,W
      label ([np.array numpy.uint8]): H,W
      model_points ([np.array numpy.float32]): 2300,3
      obj_idx: (Int)
      K_real ([np.array numpy.float32]): 3,3
      cam_flag (Bool)
      h_gt ([np.array numpy.float32]): 4,4
      h_real_est ([np.array numpy.float32]): 4,4
    Returns:
      real_img ([torch.tensor torch.float32]): H,W,3
      render_img ([torch.tensor torch.float32]): H,W,3
      real_d ([torch.tensor torch.float32]): H,W
      render_d ([torch.tensor torch.float32]): H,W
      gt_label_cropped ([torch.tensor torch.long]): H,W
      u_cropped_scaled ([torch.tensor torch.float32]): H,W
      v_cropped_scaled([torch.tensor torch.float32]): H,W
      valid_flow_mask_cropped([torch.tensor torch.bool]): H,W
      bb ([tuple]) containing torch.tensor( real_tl, dtype=torch.int32) , torch.tensor( real_br, dtype=torch.int32) , torch.tensor( ren_tl, dtype=torch.int32) , torch.tensor( ren_br, dtype=torch.int32 )         
      h_render ([torch.tensor torch.float32]): 4,4
      h_init ([torch.tensor torch.float32]): 4,4
    """ 
    h = self._h
    w = self._w

    output_h = self.output_h
    output_w = self.output_w        
    
    if not  ( h_real_est is None ): 
      h_init = h_real_est
    else:
      nt = self._cfg_d['output_cfg'].get('noise_translation', 0.02) 
      nr = self._cfg_d['output_cfg'].get('noise_rotation', 30) 
      h_init = add_noise( h_gt, nt, nr)
      
    # transform points
    pred_points = (model_points @ h_init[:3,:3].T) + h_init[:3,3]

    init_rot_wxyz = torch.tensor( re_quat( R.from_matrix(h_init[:3,:3]).as_quat(), 'xyzw') )
    idx = torch.LongTensor([int(obj_idx) - 1])

    img_ren, depth_ren, h_render = self._vm.get_closest_image_batch(
      i=idx[None], rot=init_rot_wxyz[None,:], conv='wxyz')

    # rendered data BOUNDING BOX Computation
    bb_lsd = get_bb_from_depth(depth_ren)
    b_ren = bb_lsd[0]
    tl, br = b_ren.limit_bb()
    if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b_ren.violation():
      if self.err:
        print("Violate BB in get render data for rendered bb")
      return False
    center_ren = backproject_points(
      h_render[0, :3, 3].view(1, 3), K=self.K_ren)
    center_ren = center_ren.squeeze()
    b_ren.move(-center_ren[1], -center_ren[0])
    b_ren.expand(1.1)
    b_ren.expand_to_correct_ratio(w, w)
    b_ren.move(center_ren[1], center_ren[0])
    ren_h = b_ren.height()
    ren_w = b_ren.width()
    ren_tl = b_ren.tl
    render_img = b_ren.crop(img_ren[0], scale=True, mode="bilinear",
                output_h = output_h, output_w = output_w) # Input H,W,C        
    render_d = b_ren.crop(depth_ren[0][:,:,None], scale=True, mode="nearest",
                output_h = output_h, output_w = output_w) # Input H,W,C
    
    # real data BOUNDING BOX Computation
    bb_lsd = get_bb_real_target(torch.from_numpy( pred_points[None,:,:] ), K_real[None])
    b_real = bb_lsd[0]
    tl, br = b_real.limit_bb()
    if br[0] - tl[0] < 30 or br[1] - tl[1] < 30 or b_real.violation():
      if self.err:
        print("Violate BB in get render data for real bb")
      return False
    center_real = backproject_points(
      torch.from_numpy( h_init[:3,3][None] ), K=K_real)
    center_real = center_real.squeeze()
    b_real.move(-center_real[0], -center_real[1])
    b_real.expand(1.1)
    b_real.expand_to_correct_ratio(w, w)
    b_real.move(center_real[0], center_real[1])
    real_h = b_real.height()
    real_w = b_real.width()
    real_tl = b_real.tl
    real_img = b_real.crop(torch.from_numpy(img).type(torch.float32) , 
                 scale=True, mode="bilinear",
                 output_h = output_h, output_w = output_w)
    
    real_d = b_real.crop(torch.from_numpy(depth_real[:, :,None]).type(
      torch.float32), scale=True, mode="nearest",
      output_h = output_h, output_w = output_w)
    gt_label_cropped = b_real.crop(torch.from_numpy(label[:, :, None]).type(
      torch.float32), scale=True, mode="nearest",
      output_h = output_h, output_w = output_w).type(torch.int32)
    # LGTM 
     
    flow = self._get_flow_fast(h_render[0].numpy(), h_gt, obj_idx, 
                   label, cam_flag, b_real, 
                   b_ren, K_real, depth_ren[0],
                   output_h, output_w)
    
    
    valid_flow_mask_cropped =  b_real.crop(  torch.from_numpy( flow[2][:,:,None]).type(
      torch.float32), scale=True, mode="nearest",
      output_h = output_h, output_w = output_w).type(torch.bool).numpy()   
    if flow[2].sum() < 100:
      return False
    
    u_cropped = b_real.crop( torch.from_numpy( flow[0][:,:,None] ).type(
      torch.float32), scale=True, mode="bilinear", 
      output_h = output_h, output_w = output_w).numpy()
    v_cropped =  b_real.crop(  torch.from_numpy( flow[1][:,:,None]).type(
      torch.float32), scale=True, mode="bilinear",
      output_h = output_h, output_w = output_w).numpy()

    # scale the u and v so this is not in the uncropped space !
    _grid_x, _grid_y = np.mgrid[0:output_h, 0:output_w].astype(np.float32)
    
    nr1 = np.full((output_h,output_w), float(output_w/real_w) , dtype=np.float32)
    nr2 = np.full((output_h,output_w), float(real_tl[1])  , dtype=np.float32)
    nr3 = np.full((output_h,output_w), float(ren_tl[1]) , dtype=np.float32 )
    nr4 = np.full((output_h,output_w), float(output_w/ren_w) , dtype=np.float32 )
    v_cropped_scaled = (_grid_y -((np.multiply((( np.divide( _grid_y , nr1)+nr2) +(v_cropped[:,:,0])) - nr3 , nr4))))
    
    nr1 = np.full((output_h,output_w), float( output_h/real_h) , dtype=np.float32)
    nr2 = np.full((output_h,output_w), float( real_tl[0]) , dtype=np.float32)
    nr3 = np.full((output_h,output_w), float(ren_tl[0]) , dtype=np.float32)
    nr4 = np.full((output_h,output_w), float(output_h/ren_h) , dtype=np.float32)
    u_cropped_scaled = _grid_x -(np.round(((( _grid_x /nr1)+nr2) +np.round( u_cropped[:,:,0]))-nr3)*(nr4))
      
    ls = [real_img, render_img, \
        real_d[:,:,0], render_d[:,:,0], 
        gt_label_cropped.type(torch.long)[:,:,0],
        torch.from_numpy( u_cropped_scaled[:,:] ).type(torch.float32), 
        torch.from_numpy( v_cropped_scaled[:,:]).type(torch.float32), 
        torch.from_numpy(valid_flow_mask_cropped[:,:,0]), 
        flow[-4:],
        h_render[0].type(torch.float32),
        torch.from_numpy( h_init ).type(torch.float32),
        torch.from_numpy(h_gt).type(torch.float32),
        torch.from_numpy(K_real.astype(np.float32)),
        img_ren[0], depth_ren[0]]
    
    return ls

  def _get_flow_fast(self, h_render, h_real, idx, label_img, cam, b_real, b_ren, K_real, render_d, output_h, output_w):
    m_real = copy.deepcopy(self._mesh[idx])
    m_real = transform_mesh(m_real, h_real)

    rmi_real = RayMeshIntersector(m_real)
    tl, br = b_real.limit_bb()
    rays_origin_real = self._rays_origin_real[cam]  [int(tl[0]): int(br[0]), int(tl[1]): int(br[1])]
    rays_dir_real = self._rays_dir[cam] [int(tl[0]) : int(br[0]), int(tl[1]): int(br[1])]

    real_locations, real_index_ray, real_res_mesh_id = rmi_real.intersects_location(ray_origins=np.reshape( rays_origin_real, (-1,3) ) , 
      ray_directions=np.reshape(rays_dir_real, (-1,3)),multiple_hits=False)
    
    h_real_inv = np.eye(4)
    h_real_inv[:3,:3] = h_real[:3,:3].T
    h_real_inv[:3,3] = - h_real_inv[:3,:3] @ h_real[:3,3] 
    h_trafo =h_render @ h_real_inv
    
    ren_locations = (copy.deepcopy(real_locations) @ h_trafo[:3,:3].T) + h_trafo[:3,3]
    uv_ren = backproject_points_np(ren_locations, K=self.K_ren)
    index_the_depth_map = np.round( uv_ren )
    
    
    new_tensor = render_d[ index_the_depth_map[:,0], index_the_depth_map[:,1] ] / 10000
    distance_depth_map_to_model = torch.abs( new_tensor[:] - torch.from_numpy( ren_locations[:,2])  )
    
    valid_points_for_flow = (distance_depth_map_to_model < 0.01).numpy()
    uv_real =  backproject_points_np(real_locations, K=K_real) 
    
    valid_flow_index = uv_real[valid_points_for_flow].astype(np.uint32)
    valid_flow = np.zeros( (label_img.shape[0], label_img.shape[1]) )
    valid_flow[ valid_flow_index[:,0], valid_flow_index[:,1]] = 1

    dis = uv_ren-uv_real
    uv_real = np.uint32(uv_real)
    idx_ = np.uint32(uv_real[:,0]*(self._w) + uv_real[:,1]) 


    disparity_pixels = np.zeros((self._h,self._w,2))-999
    disparity_pixels = np.reshape( disparity_pixels, (-1,2) )
    disparity_pixels[idx_] = dis
    disparity_pixels = np.reshape( disparity_pixels, (self._h,self._w,2) )
    
    u_map = disparity_pixels[:,:,0]
    v_map = disparity_pixels[:,:,1]
    u_map = fill( u_map, u_map == -999 )
    v_map = fill( v_map, v_map == -999 )

    real_tl = np.zeros( (2) )
    real_tl[0] = int(b_real.tl[0])
    real_tl[1] = int(b_real.tl[1])
    real_br = np.zeros( (2) )
    real_br[0] = int(b_real.br[0])
    real_br[1] = int(b_real.br[1])
    ren_tl = np.zeros( (2) )
    ren_tl[0] = int(b_ren.tl[0])
    ren_tl[1] = int(b_ren.tl[1])
    ren_br = np.zeros( (2) )
    ren_br[0] = int( b_ren.br[0] )
    ren_br[1] = int( b_ren.br[1] )

    f_3 = valid_flow
    f_3 *= label_img == idx
    
    return u_map, v_map, f_3, torch.tensor( real_tl, dtype=torch.int32) , torch.tensor( real_br, dtype=torch.int32) , torch.tensor( ren_tl, dtype=torch.int32) , torch.tensor( ren_br, dtype=torch.int32 ) 

  def _load_background(self):
    # if self._cfg_d['output_cfg'].get('better_background', False):
    self.background = [ '{0}/{1}/{2}'.format(self._p_ycb, b[1],b[2][0]) for b in self._batch_list]
    self.background = [ p for p in self.background if p.find('data_syn') == -1]
    # else:
    #   p = self._cfg_env['p_background']
    #   self.background = [str(p) for p in Path(p).rglob('*.jpg')]
  
  def _get_background_image(self, obj_target_index):
    # RANDOMLY SELECT IMAGE THAT DOSENT CONTATIN obj_target_index
    while True:
      p = random.choice(self.background)
      meta = scio.loadmat( p+"-meta.mat")
      obj = meta['cls_indexes'].flatten().astype(np.int32)
      if not obj_target_index in obj:
        break 
    
    img = Image.open(p+"-color.png").convert("RGB")
    w, h = img.size
    w_g, h_g = 640, 480
    if w / h < w_g / h_g:
      h = int(w * h_g / w_g)
    else:
      w = int(h * w_g / h_g)
    crop = transforms.CenterCrop((h, w))
    img = crop(img)
    img = img.resize((w_g, h_g))
    return np.array(self._trancolor_background(img))


  def get_camera(self, idx):
    if idx == 0:
      cx = 323.7872
      cy = 279.6921
      fx = 1077.836
      fy = 1078.189
    else:
      cx = 312.9869
      cy = 241.3109
      fx = 1066.778
      fy = 1067.487
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

  def _load_flow(self):
    self._load_rays_dir() 
    self._load_meshes()

    self._max_matches = self._cfg_d.get('flow_cfg', {}).get('max_matches',1500)
    self._max_iterations =  self._cfg_d.get('flow_cfg', {}).get('max_iterations',10000)
    self._grid_x, self._grid_y = np.mgrid[0:self._h, 0:self._w]

  def _load_rays_dir(self): 
    K1 = self.get_camera('data_syn/000001', K=True)
    K2 = self.get_camera('data/0068/000001',  K=True)
    
    self._rays_origin_real = []
    self._rays_origin_render = []
    self._rays_dir = []
    
    for K in [K1,K2]:
      u_cor = np.arange(0,self._h,1)
      v_cor = np.arange(0,self._w,1)
      K_inv = np.linalg.inv(K)
      rays_dir = np.zeros((self._w,self._h,3))
      nr = 0
      rays_origin_render = np.zeros((self._w,self._h,3))
      rays_origin_real = np.zeros((self._w,self._h,3))
      for u in v_cor:
        for v in u_cor:
          n = K_inv @ np.array([u,v, 1])
          #n = np.array([n[1],n[0],n[2]])
          rays_dir[u,v,:] = n * 0.6 - n * 0.25                     
          rays_origin_render[u,v,:] = n * 0.1
          rays_origin_real[u,v,:] =  n * 0.25
          nr += 1
      rays_origin_render 
      self._rays_origin_real.append( np.swapaxes(rays_origin_real,0,1) )
      self._rays_origin_render.append( np.swapaxes(rays_origin_render,0,1) )
      self._rays_dir.append( np.swapaxes( rays_dir,0,1) )

  def _load_meshes(self):
    p = self._p_ycb + '/models'
    cad_models = [str(p) for p in Path(p).rglob('*scaled.obj')] #textured
    self._mesh = {}
    for pa in cad_models:
      try:
        idx = self._name_to_idx[pa.split('/')[-2]]
        self._mesh[ idx ] = trimesh.load(pa)
      except:
        pass

  def _get_pcd_cad_models(self):
    p = self._cfg_env['p_ycb_obj']
    class_file = open(p)
    cad_paths = []
    obj_idx = 1

    name_to_idx = {}
    name_to_idx_full = {}
    while 1:
      class_input = class_file.readline()
      if not class_input:
        break
      name_to_idx_full[class_input[:-1]] = obj_idx
      if self._obj_list_fil is not None:
        if obj_idx in self._obj_list_fil:
          cad_paths.append(
            self._cfg_env['p_ycb'] + '/models/' + class_input[:-1])
          name_to_idx[class_input[:-1]] = obj_idx
      else:
        cad_paths.append(
          self._cfg_env['p_ycb'] + '/models/' + class_input[:-1])
        name_to_idx[class_input[:-1]] = obj_idx

      obj_idx += 1

    if len(cad_paths) == 0:
      raise AssertionError

    cad_dict = {}

    for path in cad_paths:
      input_file = open(
        '{0}/points.xyz'.format(path))

      cld = []
      while 1:
        input_line = input_file.readline()
        if not input_line:
          break
        input_line = input_line[:-1].split(' ')
        cld.append([float(input_line[0]), float(
          input_line[1]), float(input_line[2])])
      cad_dict[name_to_idx[path.split('/')[-1]]] = np.array(cld)
      input_file.close()

    return cad_dict, name_to_idx, name_to_idx_full

def transform_mesh(mesh, H):
  """ directly operates on mesh and does not create a copy!"""
  t = np.ones((mesh.vertices.shape[0],4)) 
  t[:,:3] = mesh.vertices
  H[:3,:3] = H[:3,:3]
  mesh.vertices = (t @ H.T)[:,:3]
  return mesh

def rel_h (h1,h2):
  'Input numpy arrays'
  from pytorch3d.transforms import so3_relative_angle
  return so3_relative_angle(torch.tensor( h1 ) [:3,:3][None], torch.tensor( h2 ) [:3,:3][None])
  
def add_noise(h, nt = 0.01, nr= 30):
  h_noise =np.eye(4)
  while  True:
    x = special_ortho_group.rvs(3)
    #_noise[:3,:3] = R.from_euler('zyx', np.random.uniform( -nr, nr, (1, 3) ) , degrees=True).as_matrix()[0]
    if abs( float( rel_h(h[:3,:3], x)/(2* float( np.math.pi) )* 360) ) < nr:
      break
  h_noise[:3,:3] = x
  h_noise[:3,3] = np.random.normal(loc=h[:3,3], scale=nt)
  
  return h_noise

def fill(data, invalid=None):
  """
  Replace the value of invalid 'data' cells (indicated by 'invalid') 
  by the value of the nearest valid data cell
  Input:
    data:    numpy array of any dimension
    invalid: a binary array of same shape as 'data'. True cells set where data
         value should be replaced.
         If None (default), use: invalid  = np.isnan(data)
  Output: 
    Return a filled array. 
  """
  if invalid is None: invalid = np.isnan(data)
  ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
  return data[tuple(ind)]

def backproject_points_np(p, fx=None, fy=None, cx=None, cy=None, K=None):
  """
  p.shape = (nr_points,xyz)
  """
  if not K is None:
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
  # true_divide
  u = ((p[:, 0] / p[:, 2]) * fx) + cx
  v = ((p[:, 1] / p[:, 2]) * fy) + cy
  return np.stack([v, u]).T  


def test():
  import os 
  import sys
  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + '/src'))
  sys.path.append(os.path.join(os.getcwd() + '/lib'))
  def load_yaml(path):
    with open(path) as file:  
      res = yaml.load(file, Loader=yaml.FullLoader)
      return res
  cfg_env = load_yaml('/home/jonfrey/PLR3/yaml/env/env_natrix_jonas.yml')



if __name__ == "__main__":
  test()