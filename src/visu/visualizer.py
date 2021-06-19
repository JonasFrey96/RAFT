import sys
import os
import random
if __name__ == "__main__":
    # load data
    os.chdir('/home/jonfrey/PLR2')
    sys.path.append('src')
    sys.path.append('src/dense_fusion')

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
import copy
import cv2
import io
from matplotlib import cm
from torchvision import transforms
import math
from math import pi
import imageio 
from skimage.morphology import convex_hull_image
from ycb.rotations import re_quat
from ycb.ycb_helper import BoundingBox

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer as DetectronVisu
from detectron2.data import DatasetCatalog, MetadataCatalog
__all__ = ['Visualizer']

def image_functionality(func):
  def wrap(*args, **kwargs):
    log = False
    if kwargs.get('method', 'def') == 'def':
      img = func(*args,**kwargs)
      log = True
    elif kwargs.get('method', 'def') == 'left':
      kwargs_clone = copy.deepcopy(kwargs)
      kwargs_clone['store'] = False
      kwargs_clone['jupyter'] = False
      res = func(*args,**kwargs_clone)
      args[0]._storage_left = res
    elif kwargs.get('method', 'def') == 'right':
      kwargs_clone = copy.deepcopy(kwargs)
      kwargs_clone['store'] = False
      kwargs_clone['jupyter'] = False
      res = func(*args,**kwargs_clone)
      args[0]._storage_right = res
      
    if args[0]._storage_right is not None and args[0]._storage_left is not None:
      img = np.concatenate( [args[0]._storage_left,  args[0]._storage_right] , axis=1 )
      args[0]._storage_left = None
      args[0]._storage_right = None
      log = True
    
    log *= not kwargs.get('not_log', False)

    if log:
      log_exp = args[0].logger is not None
      tag = kwargs.get('tag', 'TagNotDefined')
      jupyter = kwargs.get('jupyter', False)
      # Each logging call is able to override the setting that is stored in the visualizer
      if kwargs.get('store', None) is not None:
        store = kwargs['store']
      else:
        store = args[0]._store

      if kwargs.get('epoch', None) is not None:
        epoch = kwargs['epoch']
      else:
        epoch = args[0]._epoch

      # Store & Log & Display in Jupyter
      if store:
        p = os.path.join( args[0].p_visu, f'{epoch:06d}_{tag}.png')
        imageio.imwrite(p, img)
      
      if log_exp:
        H,W,C = img.shape
        ds = cv2.resize( img , dsize=(int(W/2), int(H/2)), interpolation=cv2.INTER_CUBIC)
        if args[0].logger is not None:
          try:
            # logger == neptuneai
            args[0].logger.log_image(
              log_name = tag, 
              image = np.float32( ds )/255 , 
              step=epoch)
          except:
            try:
              # logger == tensorboard
              args[0].logger.experiment.add_image(
                tag = tag, 
                img_tensor = ds, 
                global_step=epoch,
                dataformats='HWC')
            except:
              print('Tensorboard Logging and Neptune Logging failed !!!')
              pass 
        
      if jupyter:
          display( Image.fromarray(img))  
        
    return func(*args,**kwargs)
  return wrap

def backproject_points(p, fx, fy, cx, cy):
  """
  p.shape = (nr_points,xyz)
  """
  # true_divide
  u = torch.round((torch.div(p[:, 0], p[:, 2]) * fx) + cx)
  v = torch.round((torch.div(p[:, 1], p[:, 2]) * fy) + cy)

  if torch.isnan(u).any() or torch.isnan(v).any():
    u = torch.tensor(cx).unsqueeze(0)
    v = torch.tensor(cy).unsqueeze(0)
    print('Predicted z=0 for translation. u=cx, v=cy')
    # raise Exception

  return torch.stack([v, u]).T


def get_img_from_fig(fig, dpi=180):
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=dpi)
  buf.seek(0)
  img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  buf.close()
  img = cv2.imdecode(img_arr, 1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img


class Visualizer():
  def __init__(self, p_visu, logger=None, num_classes=20, epoch=0, store=True):
    self.p_visu = p_visu
    self.logger = logger

    if not os.path.exists(self.p_visu):
      os.makedirs(self.p_visu)
    
    self._epoch = epoch
    self._store = store

    jet = cm.get_cmap('jet')
    self.SEG_COLORS = (np.stack([jet(v)
    for v in np.linspace(0, 1, num_classes)]) * 255).astype(np.uint8)
    self.SEG_COLORS_BINARY = (np.stack([jet(v)
    for v in np.linspace(0, 1, 2)]) * 255).astype(np.uint8)

    self._flow_scale= 1000
    Nc = int( np.math.pi*2 * self._flow_scale)
    cmap = plt.cm.get_cmap('hsv', Nc)
    self._flow_cmap = [cmap(i) for i in range(cmap.N)]

    self._storage_left = None
    self._storage_right = None

    
    class DotDict(dict):
      """dot.notation access to dictionary attributes"""
      __getattr__ = dict.get
      __setattr__ = dict.__setitem__
      __delattr__ = dict.__delitem__

    self._meta_data =  {
        'stuff_classes':['Invalid', 'Valid'],
        'stuff_colors':[[255,89,94],
                        [138,201,38]] 
    }
    self._meta_data = DotDict(self._meta_data)


  @property
  def epoch(self):
    return self._epoch
  @epoch.setter
  def epoch(self, epoch):
    self._epoch = epoch

  @property
  def store(self):
    return self._store
  @store.setter
  def store(self, store):
    self._store = store
  
  @image_functionality
  def plot_detectron(self, img, label, **kwargs):
    # use image function to get imagae is np.array uint8
    img = self.plot_image( img, not_log=True )
    try:
      label = label.clone().cpu().numpy()
    except:
      pass
    label = label.astype(np.long) 
    detectronVisualizer = DetectronVisu( torch.from_numpy(img).type(torch.uint8), self._meta_data, scale=1)
    out = detectronVisualizer.draw_sem_seg( label, area_threshold=None, alpha=0.5).get_image()
    return out


  @image_functionality
  def plot_image(self, img,**kwargs):
    try:
      img = img.clone().cpu().numpy()
    except:
      pass
    if img.shape[2] == 3 or img.shape[2] == 4:
      pass
    elif img.shape[0] == 3 or img.shape[0] == 4: 
      img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    else:
      raise Exception('Invalid Shape')  
    if img.max() <= 1:
      img = img*255
    img = np.uint8(img)
    return img

  @image_functionality
  def flow_to_gradient(self, img, flow, mask,tl=[0,0], br=[479,639], *args, **kwargs):
    """
    img torch.tensor(h,w,3)
    flow torch.tensor(h,w,2)
    mask torch.tensor(h,w) BOOL
    call with either: 
    """
    amp = torch.norm(flow, p=2, dim=2)
    amp = amp / (torch.max(amp)+1.0e-6)  # normalize the amplitude
    dir_bin = torch.atan2(flow[:, :, 0], flow[:, :, 1])
    dir_bin[ dir_bin < 0] += 2*np.math.pi 

    dir_bin *= self._flow_scale
    dir_bin = dir_bin.type(torch.long)
    
    h,w = 480,640
    arr = np.zeros( (h,w,4), dtype=np.uint8)
    arr2 = np.zeros( (h,w,4), dtype=np.uint8)
    arr_img = np.ones( (h,w,4), dtype=np.uint8) *255
    arr_img[:,:,:3] = img

    u_list = np.uint32( np.linspace( float( tl[0] ) , float( br[0] ), num=h))[:,None].repeat(640,1).flatten()
    v_list = np.uint32( np.linspace( float( tl[1] ) , float( br[1] ), num=w))[None,:].repeat(480,0).flatten()

    u_org = np.uint32( np.linspace( 0 , h-1, num=h))[:,None].repeat(640,1).flatten()
    v_org = np.uint32( np.linspace( 0 , w-1, num=w))[None,:].repeat(480,0).flatten()
    sel1 = dir_bin.numpy()[u_org,v_org ]
    sel1[ sel1>( len(self._flow_cmap)-1) ] = len(self._flow_cmap)-1 

    m1 = (u_list < 480) * (u_list > 0) * (v_list < 640) * (u_list > 0)
    u_list = u_list[ m1 ]
    v_list = v_list[ m1 ]
    sel1 = sel1[m1]

    arr2[u_list,v_list] = np.uint8( (np.array(self._flow_cmap)*255)[sel1]  ) 
    arr = arr2
    mask = mask[:,:,None].repeat(1,1,4).type(torch.bool).numpy()
    arr_img[mask] = arr[ mask]
    return arr_img[:,:,:3] 
    
  @image_functionality
  def plot_translations(self, img, flow, mask, min_points=50,*args,**kwargs):
    """
    img torch.tensor(h,w,3)
    flow torch.tensor(h,w,2)
    mask torch.tensor(h,w) BOOL
    """
    flow = flow * \
        mask.type(torch.float32)[:, :, None].repeat(1, 1, 2)
    # flow '[+down/up-], [+right/left-]'

    def bin_dir_amplitude(flow):
      amp = torch.norm(flow, p=2, dim=2)
      amp = amp / (torch.max(amp)+1.0e-6)  # normalize the amplitude
      dir_bin = torch.atan2(flow[:, :, 0], flow[:, :, 1])
      nr_bins = 8
      bin_rad = 2 * pi / nr_bins
      dir_bin = torch.round(dir_bin / bin_rad) * bin_rad
      return dir_bin, amp

    rot_bin, amp = bin_dir_amplitude(flow)
    s = 20

    while torch.sum(mask[::s,::s]) < min_points and s > 1:
      s -= 1

    a = 2 if s > 15 else 1
    pil_img = Image.fromarray(img.numpy().astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(pil_img)
    txt = f"""Horizontal, pos right | neg left:
max = {torch.max(flow[mask][:,0])}
min = {torch.min(flow[mask][:,0])}
mean = {torch.mean(flow[mask][:,0])}
Vertical, pos down | neg up:
max = {torch.max(flow[mask][:,1])}
min = {torch.min(flow[mask][:,1])}
mean = {torch.mean(flow[mask][:,1])}"""
    draw.text((10, 60), txt, fill=(201, 45, 136, 255))
    col = (0, 255, 0)
    grey = (207, 207, 207)
    for u in range(int(flow.shape[0] / s) - 2):
      u = int(u * s)
      for v in range(int(flow.shape[1] / s) - 2):
        v = int(v * s)
        if mask[u, v] == True:
          du = round(math.cos(rot_bin[u, v])) * s / 2 * amp[u, v]
          dv = round(math.sin(rot_bin[u, v])) * s / 2 * amp[u, v]
          try:
            draw.line([(v, u), (v + dv, u + du)],
                      fill=col, width=2)
            draw.ellipse([(v - a, u - a), (v + a, u + a)],
                          outline=grey, fill=grey, width=2)
          except:
            pass
    return np.array(pil_img).astype(np.uint8)
      
  @image_functionality
  def plot_contour(self, img, points, K =None, H = None,
                    cam_cx=0, cam_cy=0, cam_fx=0, cam_fy=0,
                    trans=[[0, 0, 0]],
                    rot_mat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    thickness=2, color=(0, 255, 0),*args,**kwargs):
    """
    path := != None creats the path and store to it path/tag.png
    img:= original_image, [widht,height,RGB], torch
    points:= points of the object model [length,x,y,z]
    trans: [1,3]
    rot: [3,3]
    """
    if K is not None: 
      cam_cx = K [0,2]
      cam_cy = K [1,2] 
      cam_fx = K [0,0]
      cam_fy = K [1,1]
    if H is not None:
      rot_mat = H[:3,:3]
      trans = H[:3,3][None,:]
      if H[3,3] != 1:
        raise Exception
      if H[3,0] != 0 or H[3,1] != 0 or H[3,2] != 0:
        raise Exception

    rot_mat = np.array(rot_mat)
    trans = np.array(trans)
    img_f = copy.deepcopy(img).astype(np.uint8)
    points = np.dot(points, rot_mat.T)
    points = np.add(points, trans[0, :])
    h = img_f.shape[0]
    w = img_f.shape[1]
    acc_array = np.zeros((h, w, 1), dtype=np.uint8)

    # project pointcloud onto image
    for i in range(0, points.shape[0]):
      p_x = points[i, 0]
      p_y = points[i, 1]
      p_z = points[i, 2]
      if p_z < 1.0e-4:
        continue
      u = int(((p_x / p_z) * cam_fx) + cam_cx)
      v = int(((p_y / p_z) * cam_fy) + cam_cy)
      try:
        a = 10
        acc_array[v - a:v + a + 1, u - a:u + a + 1, 0] = 1
      except:
        pass

    kernel = np.ones((a * 2, a * 2, 1), np.uint8)
    erosion = cv2.erode(acc_array, kernel, iterations=1)

    try:  # problem cause by different cv2 version > 4.0
      contours, hierarchy = cv2.findContours(
        np.expand_dims(erosion, 2), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    except:  # version < 4.0
      _, contours, hierarchy = cv2.findContours(
        np.expand_dims(erosion, 2), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 3)

    for i in range(h):
      for j in range(w):
        if out[i, j, 1] == 255:
          img_f[i, j, :] = out[i, j, :]

    
    return img_f.astype(np.uint8)

  @image_functionality
  def plot_segmentation(self, label, *args,**kwargs):
    try:
      label = label.clone().cpu().numpy()
    except:
      pass

    if label.dtype == np.bool:
      col_map = self.SEG_COLORS_BINARY
    else:
      col_map = self.SEG_COLORS
      label = label.round()
    
    H,W = label.shape[:2]
    img = np.zeros((H,W,3), dtype=np.uint8)
    for i, color in enumerate( col_map ) :
      img[ label==i ] = color[:3]

    return img

  @image_functionality
  def plot_convex_hull(self, img, points, K, H, 
                        color=(0,255,0,255), *args,**kwargs):
      """
      img:= original_image, [widht,height,RGB]
      points:= points of the object model [length,x,y,z]
      trans: [1,3]
      rot: [3,3]
      """
      try: 
        points = points.clone().cpu().numpy()
      except:
        pass
      try: 
        H = H.clone().cpu().numpy()
      except:
        pass
      try: 
        K = K.clone().cpu().numpy()
      except:
        pass
      
      base_layer = Image.fromarray( copy.deepcopy(img) ).convert("RGBA")
      color_layer = Image.new('RGBA', base_layer.size, color=tuple( color[:3]) )
      alpha_mask = Image.new('L', base_layer.size, 0)
      
      alpha_mask_draw = ImageDraw.Draw(alpha_mask)

      target = points @ H[:3,:3].T + H[:3,3]
      pixels = np.round(  ((K @ target.T)[:2,:] /  (K @ target.T)[2,:][None,:].repeat(2,0)).T ).astype(np.long)
      _h,_w,_ = img.shape
      m = (pixels[:,0] >= w) * (pixels[:,1] >= w) * (pixels[:,1] < (_h-w-1)) * (pixels[:,0] < (_w-w-1))
      pixels = pixels[m]
      arr = np.zeros ( img.shape[:2] , dtype= np.uint8 )
      arr[ pixels[:,1], pixels[:,0] ] = 255
      convex_mask = np.uint8( convex_hull_image( arr ) ) * color[3]
      alpha_mask = Image.fromarray( convex_mask, mode='L' )
      base_layer = np.array( Image.composite(color_layer, base_layer, alpha_mask) )
      return base_layer.astype(np.uint8)
      
  @image_functionality
  def plot_estimated_pose(self, img, points, K, H, 
                          w=2, color=(0,255,0,255),*args,**kwargs):
    """
    img:= original_image, [widht,height,RGB]
    points:= points of the object model [length,x,y,z]
    trans: [1,3]
    rot: [3,3]
    """
    try: 
      points = points.clone().cpu().numpy()
    except:
      pass
    try: 
      H = H.clone().cpu().numpy()
    except:
      pass
    try: 
      K = K.clone().cpu().numpy()
    except:
      pass

    base_layer = Image.fromarray( copy.deepcopy(img) ).convert("RGBA")
    color_layer = Image.new('RGBA', base_layer.size, color=tuple( color[:3]) )
    alpha_mask = Image.new('L', base_layer.size, 0)
    alpha_mask_draw = ImageDraw.Draw(alpha_mask)

    target = points @ H[:3,:3].T + H[:3,3]
    pixels = np.round(  ((K @ target.T)[:2,:] /  (K @ target.T)[2,:][None,:].repeat(2,0)).T )
    _h,_w,_ = img.shape
    m = (pixels[:,0] >= w) * (pixels[:,1] >= w) * (pixels[:,1] < (_h-w-1)) * (pixels[:,0] < (_w-w-1))
    pixels = pixels[m]

    for u,v in pixels.tolist():
      alpha_mask_draw.ellipse( [(u - w, v - w ), (u + w + 1,v + w + 1) ], color[3])
    base_layer = np.array( Image.composite(color_layer, base_layer, alpha_mask) )
    return base_layer.astype(np.uint8)
  
  @image_functionality
  def plot_estimated_pose_on_bb(  self, img, points, tl, br,
                                  trans=[[0, 0, 0]],
                                  rot_mat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  cam_cx=0, cam_cy=0, cam_fx=0, cam_fy=0,
                                  w=2, K = None, H=None,color_code_depth=False,max_val=2, *args,**kwargs):
    """
    img:= original_image, [widht,height,RGB]
    points:= points of the object model [length,x,y,z]
    trans: [1,3]
    rot: [3,3]
    """
    if K is not None: 
      cam_cx = K [0,2]
      cam_cy = K [1,2] 
      cam_fx = K [0,0]
      cam_fy = K [1,1]
    if H is not None:
      rot_mat = H[:3,:3]
      trans = H[:3,3][None,:]
      if H[3,3] != 1:
        raise Exception
      if H[3,0] != 0 or H[3,1] != 0 or H[3,2] != 0:
        raise Exception
          
    if type(rot_mat) == list:
      rot_mat = np.array(rot_mat)
    if type(trans) == list:
      trans = np.array(trans)

    img_d = copy.deepcopy(img)
    points = np.dot(points, rot_mat.T)
    points = np.add(points, trans[0, :])
    width = int( br[1] - tl[1] )
    height = int( br[0] - tl[0] )
    off_h = int( tl[0] ) 
    off_w = int( tl[1] )
    
    for i in range(0, points.shape[0]):
      p_x = points[i, 0]
      p_y = points[i, 1]
      p_z = points[i, 2]

      u = int( (int(((p_x / p_z) * cam_fx) + cam_cx) - off_w) / width * 640 )
      v = int( (int(((p_y / p_z) * cam_fy) + cam_cy) - off_h) / height * 480 )

      try:
        if color_code_depth:
          z = min( max(0,points[i, 2]), max_val)/max_val
          turbo = cm.get_cmap('turbo', 256)
          rgba = turbo(float(z))
          img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = int( rgba[0]*255 )
          img_d[v - w:v + w + 1, u - w:u + w + 1, 1] = int( rgba[1]*255 )
          img_d[v - w:v + w + 1, u - w:u + w + 1, 2] = int( rgba[2]*255 )
        else:
          img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
          img_d[v - w:v + w + 1, u - w:u + w + 1, 1] = 255
          img_d[v - w:v + w + 1, u - w:u + w + 1, 2] = 0
      except:
        #print("out of bounce")
        pass
    
    return img_d.astype(np.uint8)

  @image_functionality
  def plot_bounding_box(self, img, rmin=0, rmax=0, cmin=0, cmax=0, str_width=2, b=None, *args,**kwargs):
    """
    img:= original_image, [widht,height,RGB]
    """

    if isinstance(b, dict):
      rmin = b['rmin']
      rmax = b['rmax']
      cmin = b['cmin']
      cmax = b['cmax']

    # ToDo check Input data
    img_d = np.array(copy.deepcopy(img))

    c = [0, 0, 255]
    rmin_mi = max(0, rmin - str_width)
    rmin_ma = min(img_d.shape[0], rmin + str_width)

    rmax_mi = max(0, rmax - str_width)
    rmax_ma = min(img_d.shape[0], rmax + str_width)

    cmin_mi = max(0, cmin - str_width)
    cmin_ma = min(img_d.shape[1], cmin + str_width)

    cmax_mi = max(0, cmax - str_width)
    cmax_ma = min(img_d.shape[1], cmax + str_width)

    img_d[rmin_mi:rmin_ma, cmin:cmax, :] = c
    img_d[rmax_mi:rmax_ma, cmin:cmax, :] = c
    img_d[rmin:rmax, cmin_mi:cmin_ma, :] = c
    img_d[rmin:rmax, cmax_mi:cmax_ma, :] = c
    img_d = img_d.astype(np.uint8)
    return img_d.astype(np.uint8)

  @image_functionality
  def plot_batch_projection(self, images, target, cam,max_images=10, *args,**kwargs):

    num = min(max_images, target.shape[0])
    fig = plt.figure(figsize=(7, num * 3.5))
    for i in range(num):
      masked_idx = backproject_points(
          target[i], fx=cam[i, 2], fy=cam[i, 3], cx=cam[i, 0], cy=cam[i, 1])

      for j in range(masked_idx.shape[0]):
        try:
          images[i, int(masked_idx[j, 0]), int(
              masked_idx[j, 1]), 0] = 0
          images[i, int(masked_idx[j, 0]), int(
              masked_idx[j, 1]), 1] = 255
          images[i, int(masked_idx[j, 0]), int(
              masked_idx[j, 1]), 2] = 0
        except:
          pass

      min1 = torch.min(masked_idx[:, 0])
      max1 = torch.max(masked_idx[:, 0])
      max2 = torch.max(masked_idx[:, 1])
      min2 = torch.min(masked_idx[:, 1])

      bb = BoundingBox(p1=torch.stack(
          [min1, min2]), p2=torch.stack([max1, max2]))

      bb_img = bb.plot(
          images[i, :, :, :3].cpu().numpy().astype(np.uint8))
      fig.add_subplot(num, 2, i * 2 + 1)
      plt.imshow(bb_img)

      fig.add_subplot(num, 2, i * 2 + 2)
      real = images[i, :, :, :3].cpu().numpy().astype(np.uint8)
      plt.imshow(real)
  
    
    img = get_img_from_fig(fig).astype(np.uint8)
    plt.close()
    return img

  @image_functionality
  def visu_network_input(self, data, max_images=10, *args,**kwargs):
    num = min(max_images, data.shape[0])
    fig = plt.figure(figsize=(7, num * 3.5))

    for i in range(num):

      n_render = f'batch{i}_render.png'
      n_real = f'batch{i}_real.png'
      real = np.transpose(
        data[i, :3, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))
      render = np.transpose(
        data[i, 3:, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))

      # plt_img(real, name=n_real, folder=folder)
      # plt_img(render, name=n_render, folder=folder)

      fig.add_subplot(num, 2, i * 2 + 1)
      plt.imshow(real)
      plt.tight_layout()
      fig.add_subplot(num, 2, i * 2 + 2)
      plt.imshow(render)
      plt.tight_layout()
  
    img = get_img_from_fig(fig).astype(np.uint8)
    plt.close()
    return  img  
  
  @image_functionality
  def plot_depthmap(self, depth, fix_max=2, *args,**kwargs):
    arr = depth.clone().cpu().numpy()
    
    arr[0,0] = 0
    arr[0,1] = fix_max
    
    w = depth.shape[0]
    h = depth.shape[1]
    
    fig = plt.figure(figsize=(6,float(w/h*6)))
    ax = []
    ax.append( fig.add_subplot(1,1,1)  )
    ax[-1].get_xaxis().set_visible(False)
    ax[-1].get_yaxis().set_visible(False)
    pos = ax[-1].imshow( arr, cmap='turbo' )
    fig.colorbar(pos, ax=ax[-1])
    
    img = get_img_from_fig(fig).astype(np.uint8)
    plt.close()
    return img
    
    
  @image_functionality
  def plot_corrospondence(self, u_map, v_map, flow_mask, real_img, render_img, 
    colorful = False, text=False, res_h =30, res_w=30, min_points=50, *args,**kwargs):
    """Plot Matching Points on Real and Render Image
    Args:
        u_map (torch.tensor dtype float): H,W 
        v_map (torch.tensor dtype float): H,W
        flow_mask (torch.tensor dtype bool): H,W
        real_img (torch.tensor dtype float): H,W,3
        render_img (torch.tensor dtype float): H,W,3
    """     
    cropped_comp = np.concatenate( [real_img.cpu().numpy(), render_img.cpu().numpy() ], axis=1).astype(np.uint8)
    cropped_comp_img = Image.fromarray(cropped_comp)
    draw = ImageDraw.Draw(cropped_comp_img)

    m = flow_mask != 0
    if text:
        txt = f"""Flow in Height:
max = {torch.max(u_map[m].type(torch.float32))}
min = {torch.min(u_map[m].type(torch.float32))}
mean = {torch.mean(u_map[m].type(torch.float32))}
Flow in Vertical:
max = {torch.max(v_map[m].type(torch.float32))}
min = {torch.min(v_map[m].type(torch.float32))}
mean = {torch.mean(v_map[m].type(torch.float32))}"""
        draw.text((10, 60), txt, fill=(201, 45, 136, 255))

    Nc = 20
    cmap = plt.cm.get_cmap('gist_rainbow', Nc)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    

    h, w = u_map.shape
    col = (0,255,0)

    while torch.sum(flow_mask[::res_h,::res_w]) < min_points and res_h > 1:
      res_w -= 1
      res_h -= 1

    for _w in range(0,w,res_w):
      for _h in range(0,h,res_h): 

          if flow_mask[_h,_w] != 0:
            try:
              delta_h = u_map[_h,_w]
              delta_w = v_map[_h,_w]
              if colorful:
                col = random.choice(cmaplist)[:3]
                col = (int( col[0]*255 ),int( col[1]*255 ),int( col[2]*255 ))
              draw.line([(int(_w), int(_h)), (int(_w + w - delta_w ), int( _h - delta_h))],
              fill=col, width=4)
            except:
                print('failed')
    
    return np.array( cropped_comp_img ).astype(np.uint8)
      
  @image_functionality
  def visu_network_input_pred(self, data, images, target, cam,max_images=10, *args,**kwargs ):
    num = min(max_images, data.shape[0])
    fig = plt.figure(figsize=(10.5, num * 3.5))

    for i in range(num):
      # real render input
      n_render = f'batch{i}_render.png'
      n_real = f'batch{i}_real.png'
      real = np.transpose(
          data[i, :3, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))
      render = np.transpose(
          data[i, 3:, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))
      fig.add_subplot(num, 3, i * 3 + 1)
      plt.imshow(real)
      plt.tight_layout()
      fig.add_subplot(num, 3, i * 3 + 2)
      plt.imshow(render)
      plt.tight_layout()

      # prediction
      masked_idx = backproject_points(
          target[i], fx=cam[i, 2], fy=cam[i, 3], cx=cam[i, 0], cy=cam[i, 1])
      for j in range(masked_idx.shape[0]):
        try:
          images[i, int(masked_idx[j, 0]), int(
              masked_idx[j, 1]), 0] = 0
          images[i, int(masked_idx[j, 0]), int(
              masked_idx[j, 1]), 1] = 255
          images[i, int(masked_idx[j, 0]), int(
              masked_idx[j, 1]), 2] = 0
        except:
          pass
      min1 = torch.min(masked_idx[:, 0])
      max1 = torch.max(masked_idx[:, 0])
      max2 = torch.max(masked_idx[:, 1])
      min2 = torch.min(masked_idx[:, 1])
      bb = BoundingBox(p1=torch.stack(
          [min1, min2]), p2=torch.stack([max1, max2]))
      bb_img = bb.plot(
          images[i, :, :, :3].cpu().numpy().astype(np.uint8))
      fig.add_subplot(num, 3, i * 3 + 3)
      plt.imshow(bb_img)
      # fig.add_subplot(num, 2, i * 2 + 4)
      # real = images[i, :, :, :3].cpu().numpy().astype(np.uint8)
      # plt.imshow(real)
    
    img = get_img_from_fig(fig).astype(np.uint8)
    plt.close()
    return img
# import k3d
# def plot_pcd(x, point_size=0.005, c='g'):
#   """[summary]
#   Args:
#       x ([type]): point_nr,3
#       point_size (float, optional): [description]. Defaults to 0.005.
#       c (str, optional): [description]. Defaults to 'g'.
#   """    
#   if c == 'b':
#       k = 245
#   elif c == 'g':
#       k = 25811000
#   elif c == 'r':
#       k = 11801000
#   elif c == 'black':
#       k = 2580
#   else:
#       k = 2580
#   colors = np.ones(x.shape[0]) * k
#   plot = k3d.plot(name='points')
#   plt_points = k3d.points(x, colors.astype(np.uint32), point_size=point_size)
#   plot += plt_points
#   plt_points.shader = '3d'
#   plot.display()


# def plot_two_pcd(x, y, point_size=0.005, c1='g', c2='r'):
#   if c1 == 'b':
#       k = 245
#   elif c1 == 'g':
#       k = 25811000
#   elif c1 == 'r':
#       k = 11801000
#   elif c1 == 'black':
#       k = 2580
#   else:
#       k = 2580

#   if c2 == 'b':
#       k2 = 245
#   elif c2 == 'g':
#       k2 = 25811000
#   elif c2 == 'r':
#       k2 = 11801000
#   elif c2 == 'black':
#       k2 = 2580
#   else:
#       k2 = 2580

#   col1 = np.ones(x.shape[0]) * k
#   col2 = np.ones(y.shape[0]) * k2
#   plot = k3d.plot(name='points')
#   plt_points = k3d.points(x, col1.astype(np.uint32), point_size=point_size)
#   plot += plt_points
#   plt_points = k3d.points(y, col2.astype(np.uint32), point_size=point_size)
#   plot += plt_points
#   plt_points.shader = '3d'
#   plot.display()