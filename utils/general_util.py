#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/10 2:00
# @Author  : jc Han
# @help    :
import sys
import numpy as  np
import torch
import sys
from datetime import datetime
import numpy as np
import random
from types import SimpleNamespace
from plyfile import PlyData
from config import config
def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, (list, tuple)):
        return [namespace_to_dict(v) for v in ns]
    else:
        return ns

def dict_to_namespace(d):
    if isinstance(d, dict):
        # 递归处理字典的每个值
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, (list, tuple)):
        # 递归处理列表/元组的每个元素
        return [dict_to_namespace(v) for v in d]
    else:
        # 基本类型（int/float/str等）直接返回
        return d

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(config.device)

def load_pcd(path):

    plydata = PlyData.read(path)
    gs_dict = {}

    gs_dict["points"] = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)
    gs_dict["points"] = gs_dict["points"]
    # gs_dict["colors"] = np.zeros((gs_dict["points"].shape[0], 3,1))
    # gs_dict["colors"][:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    # gs_dict["colors"][:, 1 ,0] = np.asarray(plydata.elements[0]["f_dc_1"])
    # gs_dict["colors"][:, 2 , 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    return gs_dict

def load_gs_attributes(path):

    plydata = PlyData.read(path)
    gs_dict = {}

    gs_dict["positions"] = np.stack((
       np.asarray(plydata.elements[0]["x"]),
       np.asarray(plydata.elements[0]["y"]),
       np.asarray(plydata.elements[0]["z"])
    ), axis=1)

    gs_dict["opacities"] = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    gs_dict["features_dc"] = np.zeros((gs_dict["positions"].shape[0], 3, 1))
    gs_dict["features_dc"][:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    gs_dict["features_dc"][:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    gs_dict["features_dc"][:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((gs_dict["positions"].shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((gs_dict["positions"].shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    gs_dict['scales'] = scales
    gs_dict['rot'] =rots
    return gs_dict


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * np.arctan2(w, 2 * fx)
    fov_y = 2 * np.arctan2(h, 2 * fy)
    return fov_x, fov_y