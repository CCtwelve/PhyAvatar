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



import cv2 as cv # Make sure to import cv2
import torch
import numpy as np

def save_images(image_path, data_dict, img_name=None, iteration=None) -> np.ndarray:
    if not data_dict:
        print("Warning: data_dict is empty. Nothing to save.")
        return

    processed_images = []

    # --- Step 1: Convert all images to NumPy arrays first ---
    # This loop processes tensors and gets them ready for padding.
    for key, value in data_dict.items():
        if not (isinstance(value, (torch.Tensor, np.ndarray)) and len(value.shape) >= 2):
            print(f"Warning: Item '{key}' is not a valid image array. Skipping.")
            continue

        img_np = value
        if isinstance(img_np, torch.Tensor):
            # PyTorch format is often (C, H, W), so permute to (H, W, C) for image processing
            if len(img_np.shape) == 3:
                img_np = img_np.permute(1, 2, 0)
            img_np = img_np.detach().cpu().numpy()

        # Normalize to 0-255 range if they are floats (0.0 to 1.0)
        if img_np.max() <= 1.0 and img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        # If image is grayscale (single channel), convert to 3-channel
        if len(img_np.shape) == 2: # This handles (H, W) case
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[-1] == 1: # This handles (H, W, 1) case
            img_np = np.repeat(img_np, 3, axis=-1)

        processed_images.append(img_np)

    if not processed_images:
        print("Warning: No valid images found to process.")
        return

    # --- Step 2: Find the maximum height among all processed images ---
    max_height = max(img.shape[0] for img in processed_images)

    # --- Step 3: Pad each image to match the max height ---
    padded_images = []
    for img_np in processed_images:
        current_height = img_np.shape[0]
        if current_height < max_height:
            # Calculate padding for top and bottom to center the image
            total_pad = max_height - current_height
            pad_top = total_pad // 2
            pad_bottom = total_pad - pad_top
            
            # Use np.pad for a clean and efficient way to pad the image
            # The format is ((top, bottom), (left, right), (channel_top, channel_bottom))
            padded_img = np.pad(img_np, ((pad_top, pad_bottom), (0, 0), (0, 0)), 
                                mode='constant', constant_values=0)
            padded_images.append(padded_img)
        else:
            # If the image is already at max height, no padding is needed
            padded_images.append(img_np)

    # --- Step 4: Combine the perfectly aligned images and save ---
    combined_img = np.concatenate(padded_images, axis=1)

    # Convert from RGB (for processing) to BGR (for OpenCV saving)
    # combined_img_bgr = cv.cvtColor(combined_img, cv.COLOR_RGB)

    output_filename = f"{image_path}/{img_name}_{iteration}.jpg"
    cv.imwrite(output_filename, combined_img)
    # print(f"Successfully saved combined image to {output_filename}") # Optional: for debugging
