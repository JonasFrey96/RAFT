from torchvision import transforms as tf
from torchvision.transforms import functional as F
import torch
import PIL
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torchvision.transforms import functional as tvF
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R

def plot(disp, n='test'):
    fig, (ax1) = plt.subplots(figsize=(6, 6), ncols=1)
    pos = ax1.imshow(disp[:,:], cmap='Blues')
    fig.colorbar(pos, ax=ax1)
    plt.savefig(f'{n}.png')

__all__ = ['Augmentation']
import random
import numpy as np

class Augmentation():
    def __init__(self, 
            output_size=(300,300), 
            add_depth = False,
            degrees = 0, 
            flip_p = 0,
            color_jitter = [0.2, 0.2, 0.2, 0.05],
            jitter_real = True, 
            jitter_render = True, 
            normalize = True,
            return_non_normalized = True,
            return_ready = True, input_size= (300,300) ):  
        self.add_depth = add_depth
        self.up_in = torch.nn.UpsamplingBilinear2d(size=output_size)
        self.up_nn_in= torch.nn.UpsamplingNearest2d(size=output_size)
        
        
        self.affine_flip =  torch.tensor( [[[-1,0,1],[0,1,1]]], 
                                         dtype=torch.float32 )
        
        self.flip_p = flip_p
        self.degrees = degrees
        H,W = input_size
        grid_x = np.linspace(0,H-1,H) 
        grid_x = np.repeat( grid_x[:,None],W, axis=1)
        grid_y = np.linspace(0,W-1,W) 
        grid_y = np.repeat( grid_y[None,:],H, axis=0)
        self.grid_xy = np.stack( [grid_y, grid_x],axis=2)
        
        self.return_non_normalized = return_non_normalized 
        self.jitter_real = jitter_real
        self.jitter_render = jitter_render
        self.normalize = normalize
        self.return_ready = return_ready
        if jitter_real or jitter_render:
            self._jitter = tf.ColorJitter(*color_jitter)
            
        if normalize:
            self._norm = tf.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        
    def apply(self, idx, u_map, v_map, flow_mask, gt_label_cropped, real_img, render_img, real_d, render_d):
        """[summary]

        Parameters
        ----------
        idx : torch.tensor id
        u_map : HxW
        v_map : HxW
        flow_mask : bool HxW
        gt_label_cropped : int64 HxW
        real_img : HWC 0-255
        render_img : HWC 0-255
        """
        
        render_img = render_img/255.0
        real_img = real_img/255.0
        gt_sel = (idx.repeat(1,*gt_label_cropped.shape[0:])+1)[0]
        obj_mask = gt_label_cropped == gt_sel
        flow_mask = flow_mask * obj_mask
        
        inp  = ( u_map, v_map, flow_mask, gt_label_cropped, real_img, real_d)
        if random.random() < self.flip_p:
            
            inp = self.affine_grid(self.affine_flip, *inp)
        
        # affine = self.get_affine()
        # inp = self.affine_grid( affine, *inp)
        
        render_img = render_img.permute(2,0,1) 
        if self.jitter_real:
            real_img  = self._jitter( inp[4].permute(2,0,1)  )
        if self.jitter_render:
            render_img = self._jitter( render_img ) # C,H,W

        if self.normalize:
            if self.return_non_normalized:
                non_norm_real_img = real_img.clone() 
                non_norm_render_img = render_img.clone()
                 
            real_img = self._norm( real_img )
            render_img = self._norm( render_img )  # C,H,W
        
        if self.return_ready:
            data = torch.cat([real_img, render_img], dim=0)
            data = self.up_in(data[None])[0] 
            
            if self.add_depth:
                d = torch.stack([inp[-1], render_d], dim=0)
                d = torch.clamp(d, 0,20000)
                d = self.up_nn_in(d[None])[0] / 10000
                data = torch.cat([data,d],dim=0)
                
            # self.up_nn_in( [], dim=0 )
            
            uv = torch.stack([u_map, v_map], dim=0)  # C,H,W
            
            flow_mask = inp[2]
            flow_mask = flow_mask[None,:,:].repeat(2,1,1)
            
            gt_label_cropped = inp[3]
            return data, uv, flow_mask, gt_label_cropped, non_norm_real_img, non_norm_render_img
    
    def get_affine(self):
        angle = (random.random()-0.5) * self.degrees * 2
        rin = R.from_euler('z', angle, degrees=True).as_matrix()
        affine = torch.ones( (1,2,3) )
        affine[:,:2,:2] = torch.tensor( rin )[:2,:2] 
        return affine   
                        
    def affine_grid(self, affine, u_map, v_map, flow_mask, gt_label_cropped, real_img, real_d):
        N = 1
        C = 2
        H = 480
        W = 640

        grid = F.affine_grid(affine, (N,C,H,W),align_corners=True)
        grid -= 1
        
        real_img_rotate = F.grid_sample((real_img[None,:,:,:]).permute(0,3,1,2).type(torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        mask_rotate = F.grid_sample((flow_mask[None,:,:,None].type(torch.float32)).permute(0,3,1,2), grid, mode='nearest', padding_mode='zeros', align_corners=True)
        gt_label_cropped_rotated = F.grid_sample((gt_label_cropped[None,:,:,None].type(torch.float32)).permute(0,3,1,2), grid, mode='nearest', padding_mode='zeros', align_corners=True)
        
        real_d_rotated = F.grid_sample((real_d[None,:,:,None].type(torch.float32)).permute(0,3,1,2), grid, mode='nearest', padding_mode='zeros', align_corners=True)[0,0,:,:]
        
        gt_label_cropped_rotated = gt_label_cropped_rotated[0,0,:,:]
            
        real_img_rotate = real_img_rotate.permute(2,3,1,0)[:,:,:,0]
        grid[:,:,:,0] = (grid[:,:,:,0]+1)/2*(W-1)
        grid[:,:,:,1] = (grid[:,:,:,1]+1)/2*(H-1)

        disp = self.grid_xy - grid.numpy()[0]

        # uu = u_rotate[0,0] + disp[:,:,1] 
        # vv = v_rotate[0,0] + disp[:,:,0] 

        u_map = u_map + disp[:,:,1] 
        v_map = v_map + disp[:,:,0] 
        u_rotate = F.grid_sample((u_map[None,:,:,None]).permute(0,3,1,2).type(torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        v_rotate = F.grid_sample((v_map[None,:,:,None]).permute(0,3,1,2).type(torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        uu = u_rotate[0,0]
        vv = v_rotate[0,0]
        flow_mask_out = (mask_rotate == 1)[0,0]

        return ( uu, vv, flow_mask_out, gt_label_cropped_rotated, real_img_rotate, real_d_rotated)