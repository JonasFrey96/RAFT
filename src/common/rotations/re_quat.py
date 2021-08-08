import torch
from torch import nn

def re_quat(q, input_format):
    assert torch.is_tensor(q)
    if len( q.shape ) == 1:
        q = q[None] 
    assert q.shape[1] == 4
    p = q.clone()
    if input_format == 'xyzw':
            p[:, 0] = q[:, 3]
            p[:, 3] = q[:, 2]
            p[:, 2] = q[:, 1]
            p[:, 1] = q[:, 0]
    elif input_format == 'wxyz':
            p[:, 0] = q[:, 1]
            p[:, 1] = q[:, 2]
            p[:, 2] = q[:, 3]
            p[:, 3] = q[:, 0]
    else:
        raise Exception("Invalid Input")
    return p

if __name__ == "__main__":
    bs = 10
    from scipy.spatial.transform import Rotation as R
    from scipy.stats import special_ortho_group
    mat = special_ortho_group.rvs(dim=3, size=bs)
    quat = R.from_matrix(mat).as_quat()

    q = torch.from_numpy(quat)
    print('input', q)
    res = re_quat(q, input_format='xyzw')

    print('Output', res)
    res2 = re_quat(res, input_format='wxyz')

    print('Same as Input', res2)
