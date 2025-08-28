#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/17 16:32
# @Author  : jc Han
# @help    :
import torch

def _load_pose(pose):
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    c2w = torch.tensor([[0.9999599793, -0.0087304475, 0.0019542233, 0.0052683778],
                        [-0.0082560056, -0.9846474371, -0.1743595794, 1.4479724903],
                        [0.0034464581, 0.1743364673, -0.9846801095, 2.4680413621],
                        [0., 0., 0., 1.]])
    c2w = c2w.float()

    return c2w, intrinsic