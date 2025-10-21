#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network.lpips import LPIPS
from fused_ssim import fused_ssim

from config import config

try:
    import wandb
except Exception:
    wandb = None
try:
    import yaml
except Exception:
    yaml = None
lpips = LPIPS(net="vgg").to(config.device)
for p in lpips.parameters():
    p.requires_grad = False


import torch
import torch.nn as nn
from collections import deque
import numpy as np
import segmentation_models_pytorch as smp   

from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim_calculator = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none')

class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        # Create the loss function objects ONCE here
        self.bce_loss = nn.BCELoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, gt):
        # First, calculate the individual loss VALUES
        bce = self.bce_loss(pred, gt)
        dice = self.dice_loss(pred, gt)
        
        # Then, combine the resulting VALUES
        combined_loss = self.weight_bce * bce + self.weight_dice * dice
        
        return combined_loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


# Ellipsoid Regularization
def ellipsoid_regularization(gs, aniso_thres, max_scale,iterations,densify_until_iter):
    gaussian_scaling = gs.get_scaling

    aniso = torch.max(gaussian_scaling, dim=1)[0] / (
        torch.min(gaussian_scaling, dim=1)[0] + 1e-5
    )
    aniso_clip = torch.clamp(aniso, aniso_thres, None) - aniso_thres
    loss_aniso = torch.mean(aniso_clip)
    if iterations  > densify_until_iter:
        return torch.tensor(0.0, device=config.device)
    else:
        return loss_aniso 

import torch
import torch.nn as nn

def total_variation_loss(img):
    """
    计算全变分损失 (Total Variation Loss)。
    
    参数:
    img (torch.Tensor): 输入的图像张量，形状应为 (B, C, H, W)，
                        其中 B=批大小, C=通道数, H=高度, W=宽度。
                        对于 mask，通道数 C 通常为 1。
                        
    返回:
    torch.Tensor: 一个标量张量，表示该批次图像的平均 TV Loss。
    """
    # 计算水平方向上的差异
    # 取所有像素，除了最后一列
    pixels_except_last_col = img[:, :, :, :-1]
    # 取所有像素，除了第一列
    pixels_except_first_col = img[:, :, :, 1:]
    # 计算水平梯度
    horizontal_variation = torch.abs(pixels_except_last_col - pixels_except_first_col)
    
    # 计算垂直方向上的差异
    # 取所有像素，除了最后一行
    pixels_except_last_row = img[:, :, :-1, :]
    # 取所有像素，除了第一行
    pixels_except_first_row = img[:, :, 1:, :]
    # 计算垂直梯度
    vertical_variation = torch.abs(pixels_except_last_row - pixels_except_first_row)
    
    # 对所有差异求和，然后除以批次大小，得到平均 TV Loss
    # 您也可以选择求均值 torch.mean()，这在调整权重时可能更稳定
    tv_loss = torch.sum(horizontal_variation) + torch.sum(vertical_variation)
    
    # 标准化：除以批次大小
    b, c, h, w = img.shape
    return tv_loss / b

def scale_regularization_loss(gaussians, scene_extent, lambda_scale_reg, percent_dense,densify_until_iter,iterations):

    # 1. 获取所有高斯球的最大尺度
    max_scaling = torch.max(gaussians.get_scaling, dim=1).values
    
    # 2. 计算目标尺度阈值 0.01 * 1
    target_scaling = percent_dense * scene_extent
    
    # 3. 计算超出阈值的部分
    if iterations < densify_until_iter:
        excess_scaling = F.relu(max_scaling - target_scaling)

        scale_loss = lambda_scale_reg * torch.mean(excess_scaling)
    else:
        excess_upper = F.relu(max_scaling - (percent_dense + 0.03) *scene_extent)
        excess_lower = F.relu((percent_dense - 0.01) *scene_extent - max_scaling)
        scale_loss = lambda_scale_reg * torch.mean(excess_upper + excess_lower)
    
    return scale_loss

def compute_quadratic_sparsity_loss(gaussians):
    opacities = gaussians.get_opacity

    # o * (1-o) 在 o=0.5 时最大，在 o=0 或 o=1 时为0
    return torch.mean(opacities * (1.0 - opacities))

def compute_lpips_loss(image, gt_image):
    # assert image.shape[1] == image.shape[2] and gt_image.shape[1] == gt_image.shape[2]

    lpips_loss = lpips.forward(
        image[None, [2, 1, 0]], gt_image[None, [2, 1, 0]], normalize=True
    ).mean()
    return lpips_loss

def compute_masked_lpips_loss(
    lpips_fn: torch.nn.Module,
    render_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    gt_mask: torch.Tensor
) -> torch.Tensor:
    """
    正确计算带有掩码的 LPIPS 损失。

    Args:
        lpips_fn: LPIPS 网络实例 (必须在初始化时设置了 spatial=True)。
        render_rgb: 渲染出的图像张量，形状 [B, 3, H, W]，数值范围 [0, 1]。
        gt_image: 真值图像张量，形状 [B, 3, H, W]，数值范围 [0, 1]。
        gt_mask: 真值掩码张量，形状 [B, 1, H, W]，前景为 1.0，背景为 0.0。

    Returns:
        一个标量 (scalar) 张量，代表最终的 LPIPS 损失。
    """
    # 1. 准备输入：LPIPS VGG 网络需要 [-1, 1] 范围的 RGB 图像。
    #    这个标准化步骤必须在 *完整* 图像上进行，*不是* 在掩码后的图像上！
    #    (如果你的输入已经是 [-1, 1]，可以跳过此步)
    render_norm = render_rgb * 2.0 - 1.0
    gt_norm = gt_image * 2.0 - 1.0

    # 2. 计算 *未经平均* 的空间损失图。
    #    因为 lpips_fn 初始化时设置了 spatial=True, 
    #    所以 forward() 会返回一个形状为 [B, 1, H_out, W_out] 的损失图。
    lpips_loss_map = lpips_fn.forward(render_norm, gt_norm)

    # 3. (关键!) 将你的 gt_mask '下采样' 到 LPIPS 损失图的分辨率。
    #    获取 LPIPS 特征图的输出尺寸 H_out, W_out。
    _, _, H_out, W_out = lpips_loss_map.shape
    
    #    使用 'nearest' (最近邻) 模式来下采样 mask, 以保持 0/1 的特性。
    downsampled_mask = F.interpolate(gt_mask, size=(H_out, W_out), mode='nearest')

    # 4. (关键!) 应用 mask 并使用 sum() / sum() 计算“掩码均值”。
    #    这可以确保损失不受 mask 大小的影响。
    masked_lpips_loss_map = lpips_loss_map * downsampled_mask
    
    #    计算分母：mask 中非零元素的总数。
    num_fg_elements_lpips = downsampled_mask.sum() + 1e-8
    
    #    计算最终的 loss。
    lpips_loss = masked_lpips_loss_map.sum() / num_fg_elements_lpips
    
    return lpips_loss

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def refine_total_loss(args,iterations,tps, render_alpha, gt_mask, render_rgb, gt_image, gaussians, total_mask_loss_fn,lpips_net,extent,percent_dense):
    if getattr(tps.loss, "use_invmask_rgb", False):
        loss_rm =F.l1_loss(render_rgb, gt_image)
    else:
        l1_rgb = torch.abs(render_rgb - gt_image)
        l1_rgb_masked = l1_rgb * gt_mask
        num_fg_elements = gt_mask.sum() * render_rgb.shape[0] + 1e-8
        loss_rm = l1_rgb_masked.sum() / num_fg_elements

    if getattr(tps.loss, "lamda_tv", 0)!= 0 and iterations > args.OptimizationParams.densify_until_iter:
        tv_loss = total_variation_loss(render_alpha.unsqueeze(0))
        loss_tv = tps.loss.lamda_tv * tv_loss
    else:
        loss_tv = torch.tensor(0.0, device=config.device)
    

    if iterations < args.OptimizationParams.densify_until_iter:
        loss_am =tps.loss.lamda_l1mask *  F.l1_loss(render_alpha, gt_mask)
    else:
        if getattr(tps.loss, "lamda_l2mask",0)!=0:
            loss_am =tps.loss.lamda_l2mask *  l2_loss(render_alpha, gt_mask)
        else:
            if getattr(tps.loss, "lamda_l1mask", 0)!= 0:
                if getattr(tps.loss, "use_invmask", False):
                    inv_mask = 1.0 - gt_mask
                    leaked_alpha = render_alpha * inv_mask
                    loss_leaked_alpha = leaked_alpha.mean()
                    loss_mask_l1 = F.l1_loss(render_alpha, gt_mask)
                    loss_am = tps.loss.lamda_l1mask * (0.8 * loss_mask_l1 + 0.2 * loss_leaked_alpha)
                else:
                    loss_am =tps.loss.lamda_l1mask *  F.l1_loss(render_alpha, gt_mask)
            else:   
                loss_am = torch.tensor(0.0, device=config.device)

    ssim_map = ssim_calculator(render_rgb.unsqueeze(0), gt_image.unsqueeze(0)) 
    ssim_loss_map  = 1.0 - ssim_map

    masked_ssim_loss_map = ssim_loss_map * gt_mask
    num_fg_elements_ssim = gt_mask.sum() + 1e-8 

    ssim_loss = tps.loss.lamda_ssim * (masked_ssim_loss_map.sum() / num_fg_elements_ssim)

    lpips_loss = tps.loss.lamda_lpips * compute_masked_lpips_loss(lpips_net,render_rgb.unsqueeze(0),gt_image.unsqueeze(0),gt_mask.unsqueeze(0))
    
    ellipsoid_loss = tps.loss.lamda_ellipsoid * ellipsoid_regularization(
        gaussians, tps.aniso_thres, tps.max_scale,iterations,args.OptimizationParams.densify_until_iter
    )
        
    scale_loss = tps.loss.lamda_scale * scale_regularization_loss(gaussians, extent, tps.loss.lamda_scale, percent_dense,args.OptimizationParams.densify_until_iter , iterations)
    dict_loss = {
        "L1": loss_rm,
        "ellipsoid_loss":ellipsoid_loss,
        "lpips_loss":  lpips_loss,
        "ssim":  ssim_loss,
        "l1_mask_loss":  loss_am,
        "tv_loss":  loss_tv,
        "combined_mask_loss": torch.tensor(0.0, device=config.device),
        "scale_loss": scale_loss,
    }

    lamda_opacity = getattr(tps.loss, "lamda_opacity", 0)
    
    if lamda_opacity != 0:
        opacity_loss = compute_quadratic_sparsity_loss(gaussians)
        dict_loss["opacity_loss"] = tps.loss.lamda_opacity * opacity_loss
    else:
        dict_loss["opacity_loss"] = torch.tensor(0.0, device=config.device)

    # lamda_L1_mask = getattr(tps.loss, "lamda_L1_mask", 0)
    # lamda_combined_mask = getattr(tps.loss, "lamda_combined_mask", 0)

    # assert (lamda_L1_mask == 0) or (lamda_combined_mask == 0), \
    #     "lamda_L1_mask and lamda_combined_mask cannot both be non-zero."

    # dict_loss["l1_mask_loss"] = torch.tensor(0.0, device=config.device)
    # dict_loss["combined_mask_loss"] = torch.tensor(0.0, device=config.device)
    
    # if lamda_combined_mask != 0:
    #     combined_mask_loss = total_mask_loss_fn(render_alpha, gt_mask)
    #     dict_loss["combined_mask_loss"] = tps.loss.lamda_combined_mask * combined_mask_loss
    # elif lamda_L1_mask != 0:
    #     l1_mask_loss = l1_loss(render_alpha.float(), gt_mask.float())
    #     dict_loss["l1_mask_loss"] = tps.loss.lamda_L1_mask * l1_mask_loss
        
    loss = sum(dict_loss.values())
    dict_loss["total_loss"] = loss

    return loss, dict_loss

def total_loss(tps, render_alpha, gt_mask, render_rgb, gt_image, gaussians, total_mask_loss_fn):

    mask_render_rgb = render_rgb * gt_mask
    mask_gt_image = gt_image * gt_mask

    if tps.loss.gt_mask:
        L1 = l1_loss(mask_render_rgb, mask_gt_image)
    else:
        L1 = l1_loss(render_rgb, gt_image)
    # print("--------------------------",mask_render_rgb.shape, mask_gt_image.shape) torch.Size([1, 1022, 747]) torch.Size([3, 1022, 747])
    lpips_loss = compute_lpips_loss(mask_render_rgb, mask_gt_image)

    # ellipsoid_loss = ellipsoid_regularization(
    #     gaussians, tps.aniso_thres, tps.max_scale
    # )

    # ssim_value = fused_ssim(render_rgb.unsqueeze(0), gt_image.unsqueeze(0))
    ssim_loss = 1.0 - ssim(render_rgb, gt_image)

    dict_loss = {
        "L1": tps.loss.lamda_L1 * L1,
        "ellipsoid_loss": tps.loss.lamda_ellipsoid * torch.tensor(0.0, device=config.device),
        "lpips_loss": tps.loss.lamda_lpips * lpips_loss,
        "ssim": tps.loss.lamda_ssim * ssim_loss,
        "l1_mask_loss": torch.tensor(0.0, device=config.device),
        "combined_mask_loss": torch.tensor(0.0, device=config.device)
    }
    lamda_opacity = getattr(tps.loss, "lamda_opacity", 0)
    
    if lamda_opacity != 0:
        opacity_loss = compute_quadratic_sparsity_loss(gaussians)
        dict_loss["opacity_loss"] = tps.loss.lamda_opacity * opacity_loss
    else:
        dict_loss["opacity_loss"] = torch.tensor(0.0, device=config.device)

    lamda_L1_mask = getattr(tps.loss, "lamda_L1_mask", 0)
    lamda_combined_mask = getattr(tps.loss, "lamda_combined_mask", 0)

    assert (lamda_L1_mask == 0) or (lamda_combined_mask == 0), \
        "lamda_L1_mask and lamda_combined_mask cannot both be non-zero."
    
    if lamda_combined_mask != 0:
        combined_mask_loss = total_mask_loss_fn(render_alpha, gt_mask)
        dict_loss["combined_mask_loss"] = tps.loss.lamda_combined_mask * combined_mask_loss
    elif lamda_L1_mask != 0:
        l1_mask_loss = l1_loss(1- render_alpha.float(), 1- gt_mask.float())
        dict_loss["l1_mask_loss"] = tps.loss.lamda_L1_mask * l1_mask_loss
        
    loss = sum(dict_loss.values())
    dict_loss["total_loss"] = loss

    return loss, dict_loss



def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_ema_losses_and_string(dict_loss, iteration):
    """
    Calculate EMA losses for each component and generate a formatted string
    
    Parameters:
    dict_loss: dict, current loss values
    iteration: int, current iteration number
    
    Returns:
    str: formatted loss string with EMA values
    """
    # Initialize EMA losses for each component
    ema_losses = {}
    
    # Calculate EMA for each loss component
    for loss_name, loss_value in dict_loss.items():
        if isinstance(loss_value, torch.Tensor):
            loss_val = loss_value.item()
        else:
            loss_val = loss_value
        
        # Initialize EMA for this loss component if not exists
        if loss_name not in ema_losses:
            ema_losses[loss_name] = loss_val
        else:
            # Apply EMA smoothing: 0.4 * current + 0.6 * previous
            ema_losses[loss_name] = 0.4 * loss_val + 0.6 * ema_losses[loss_name]
    
    # Create a string with all smoothed loss components
    loss_str = f"[ITER {iteration}] "
    for loss_name, loss_value in dict_loss.items():
        loss_str += f"{loss_name}: {ema_losses[loss_name]:.7f} "
    
    return loss_str

