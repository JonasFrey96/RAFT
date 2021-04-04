import torch

def so3_relative_angle(R1, R2, cos_angle: bool = False):
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle)

def so3_rotation_angle(R, eps: float = 1e-4, cos_angle: bool = False):
    N, dim1, dim2 = R.shape
    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    rot_trace = torch.clamp(rot_trace, -1.0, 3.0)
    phi = 0.5 * (rot_trace - 1.0)
    if cos_angle:
        return phi
    else:
        return phi.acos()