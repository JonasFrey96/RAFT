if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    sys.path.append(os.path.join(os.getcwd() + '/src'))
    sys.path.append(os.path.join(os.getcwd() + '/lib'))

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from ycb.rotations import quat_to_rot
import copy
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.transform import Rotation as R
from ycb.rotations import rot_to_quat
def knn(ref, query):
    """return indices of ref for each query point. L2 norm

    Args:
        ref ([type]): points * 3
        query ([type]): tar_points * 3

    Returns:
        [knn]: distance = query * 1 , indices = query * 1
    """
    mp2 = ref.unsqueeze(0).repeat(query.shape[0], 1, 1)
    tp2 = query.unsqueeze(1).repeat(1, ref.shape[0], 1)
    dist = torch.norm(mp2 - tp2, dim=2, p=None)
    knn = dist.topk(1, largest=False)
    return knn


def loss_calculation_add(target, model_points, idx, sym_list, pred_r = None, pred_t = None, H= None):
    """ADD loss calculation

    Args:
        pred_r ([type]): BS * 3
        pred_t ([type]): BS * 4 'wxyz'
        idx ([type]): BS * 1

        model_points ([type]): BS * num_points * 3 : randomly selected points of the CAD model
        target ([type]): BS * num_points * 3 : model_points rotated and translated according to the regression goal (not ground truth because of data augmentation)

        sym_list ([list of integers]):
    Returns:
        [type]: [description]
    """
    bs, num_p, _ = target.shape
    num_point_mesh = num_p

    if H is not None:
        base = H[:,:3,:3].permute(0, 2, 1)
        pred_t = H[:,:3,3].unsqueeze(1)
    else:
        pred_r = pred_r / torch.norm(pred_r, dim=1).view(bs, 1)
        base = quat_to_rot(pred_r, 'wxyz').unsqueeze(1)
        base = base.view(-1, 3, 3).permute(0, 2, 1)  # transposed of R
        pred_t = pred_t.unsqueeze(1)

    pred = torch.add(torch.bmm(model_points, base), pred_t)
    tf_model_points = pred.view(target.shape)

    for i in range(bs):
        # ckeck if add-s or add
        if idx[i, 0].item() in sym_list:

            knn_obj = knn(
                ref=target[i, :, :], query=tf_model_points[i, :, :])
            inds = knn_obj.indices
            target[i, :, :] = target[i, inds[:, 0], :]

    dis = torch.mean(torch.norm((tf_model_points - target), dim=2), dim=1)
    return dis


class AddSLoss(nn.Module):

    def __init__(self, sym_list):
        super(AddSLoss, self).__init__()
        self.sym_list = sym_list

    def forward(self, target, model_points, idx, pred_r=None, pred_t=None, H=None):
        if H is not None:
            # pred_t = H[:,:3,3]
            # pred_r = rot_to_quat( H[:,:3,:3], "wxyz" )
            return loss_calculation_add(target.clone(), model_points.clone(), idx, self.sym_list, H = H)
        else:
            return loss_calculation_add(target.clone(), model_points.clone(), idx, self.sym_list, pred_r = pred_r, pred_t =pred_t)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    from scipy.stats import special_ortho_group
    from helper import re_quat
    from deep_im import RearangeQuat

    device = 'cuda:0'
    bs = 100
    nr_points = 3000

    re_q = RearangeQuat(bs)
    mat = special_ortho_group.rvs(dim=3, size=bs)
    quat = R.from_matrix(mat).as_quat()
    q = torch.from_numpy(quat.astype(np.float32)).cuda()
    re_q(q, input_format='xyzw')
    pred_r = q.unsqueeze(0)

    pred_t_zeros = torch.zeros((bs, 3), device=device)
    pred_t_ones = torch.ones((bs, 3), device=device)

    model_points = torch.rand((bs, nr_points, 3), device=device)
    res = 0.15
    target_points = model_points + \
        torch.ones((bs, nr_points, 3), device=device) * \
        float(np.sqrt(res * res / 3))
    pred_r_unit = torch.zeros((bs, 4), device=device)
    pred_r_unit[:, 0] = 1

    sym_list = [0]
    loss_add = AddSLoss(sym_list=sym_list)

    idx_sym = torch.zeros((bs, 1), device=device)
    idx_nonsym = torch.ones((bs, 1), device=device)

    points = target_points
    num_pt_mesh = nr_points

    loss = loss_add(pred_r_unit, pred_t_zeros,
                    target_points, model_points, idx_nonsym)
    print(f'dis = {loss} should be {res}')

    # random shuffle index
    rand_index = torch.randperm(nr_points)
    target_points = model_points[:, rand_index, :]
    loss = loss_add(pred_r_unit, pred_t_zeros,
                    target_points, model_points, idx_nonsym)

    loss = loss_add(pred_r_unit, pred_t_zeros,
                    target_points, model_points, idx_sym)
