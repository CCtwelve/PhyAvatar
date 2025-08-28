#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/8 0:15
# @Author  : jc Han
# @help    :
import  torch
import numpy as np
import torch
from typing import Dict, Any
import cv2 as cv
from PIL import Image
import os
def common_world(A_w2c,B_w2c):
    A = torch.from_numpy(A_w2c).float()
    B = torch.from_numpy(B_w2c).float()
    identity = torch.eye(4, dtype=torch.float32)

    if torch.allclose(A @ torch.inverse(B), identity, atol=1e-6):
        print("两相机共享世界坐标系")
    else:
        print("世界坐标系未对齐，需转换")


from types import SimpleNamespace

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



def save_images(image_path,data_dict,img_name = None, iteration=None,) -> np.ndarray:
    images = []

    for key, value in data_dict.items():


        if not (isinstance(value, (torch.Tensor, np.ndarray)) and len(value.shape) >= 2):
            print("data error")
            continue

        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value = value.permute(1, 2, 0).detach().cpu()
            img_np = value.detach().cpu().numpy()

        if img_np.max() <= 1.0:
            img_np  = ( img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1:  # 如果是单通道
            img_np = np.repeat(img_np, 3, axis=-1)  # 复制为三通道

        if img_np.shape[0] != 1022  or img_np.shape[1] != 747 :
            # 创建新图像（RGB模式，黑色背景）
            padded_img = np.zeros((1022, 747,3), dtype=img_np.dtype)

            start_x = (747 - img_np.shape[1]) // 2
            start_y = (1022 - img_np.shape[0]) // 2

            padded_img[start_y:start_y + img_np.shape[0],
            start_x:start_x + img_np.shape[1]] = img_np
            img_np = padded_img

        images.append(img_np)

    combined_img = np.concatenate(images, axis=1)

    cv.imwrite(f"{image_path}/{img_name}_{iteration}.jpg", combined_img)

