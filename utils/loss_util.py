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

C1 = 0.01**2
C2 = 0.03**2


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


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
def ellipsoid_regularization(gs, aniso_thres, max_scale):
    gaussian_scaling = gs.get_scaling

    aniso = torch.max(gaussian_scaling, dim=1)[0] / (
        torch.min(gaussian_scaling, dim=1)[0] + 1e-5
    )
    aniso_clip = torch.clamp(aniso, aniso_thres, None) - aniso_thres
    loss_aniso = torch.mean(aniso_clip)

    scale_penalty = torch.mean(gaussian_scaling**2)

    # print(loss_aniso,scale_penalty)

    return loss_aniso + scale_penalty


def compute_lpips_loss(image, gt_image):
    # assert image.shape[1] == image.shape[2] and gt_image.shape[1] == gt_image.shape[2]

    lpips_loss = lpips.forward(
        image[None, [2, 1, 0]], gt_image[None, [2, 1, 0]], normalize=True
    ).mean()
    return lpips_loss


def total_loss(comp_mask, gt_mask, comp_rgb, gt_image, gaussians, args):
    if args.train.loss.gt_mask:
        gt_image = gt_image * gt_mask

        comp_rgb = comp_rgb * gt_mask

    L1 = l1_loss(comp_rgb, gt_image)

    lpips_loss = compute_lpips_loss(comp_rgb, gt_image)

    ellipsoid_loss = ellipsoid_regularization(
        gaussians, args.train.aniso_thres, args.train.max_scale
    )

    L1_mask = l1_loss(comp_mask.squeeze(dim=0).float(), gt_mask.float())

    # ssim_value = fused_ssim(comp_rgb.unsqueeze(0), gt_image.unsqueeze(0))
    Lssim = 1.0 - ssim(comp_rgb, gt_image)

    loss = (
        args.train.loss.lamda_L1 * L1
        + args.train.loss.lamda_ellipsoid * ellipsoid_loss
        + args.train.loss.lamda_lpips * lpips_loss
        + args.train.loss.lamda_L1_mask * L1_mask
        + args.train.loss.lamda_ssim * Lssim
    )

    dict_loss = {
        "L1": L1,
        "ellipsoid_loss": ellipsoid_loss,
        "lpips_loss": lpips_loss,
        "L1_mask": L1_mask,
        "Lssim": Lssim,
    }
    return loss, dict_loss


def vec_to_covariance(covs_vec):
    """将 [N,6] 向量转换为 [N,3,3] 对称协方差矩阵"""
    xx, xy, xz, yy, yz, zz = covs_vec.unbind(dim=1)
    covs = torch.zeros(covs_vec.shape[0], 3, 3, device=covs_vec.device)
    covs[:, 0, 0] = xx
    covs[:, 0, 1] = covs[:, 1, 0] = xy
    covs[:, 0, 2] = covs[:, 2, 0] = xz
    covs[:, 1, 1] = yy
    covs[:, 1, 2] = covs[:, 2, 1] = yz
    covs[:, 2, 2] = zz
    return covs


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


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
