#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 6:19
# @Author  : jc Han
# @help    :
import torch
import torchvision
import numpy as np
import cv2 as cv
# def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
#     if tb_writer:
#         tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
#         tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
#         tb_writer.add_scalar('iter_time', elapsed, iteration)
#
#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
#                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
#
#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
#                 for idx, viewpoint in enumerate(config['cameras']):
#                     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
#                     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
#                     if train_test_exp:
#                         image = image[..., image.shape[-1] // 2:]
#                         gt_image = gt_image[..., gt_image.shape[-1] // 2:]
#                     if tb_writer and (idx < 5):
#                         tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                         if iteration == testing_iterations[0]:
#                             tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()
#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])
#                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
#                 if tb_writer:
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
#
#         if tb_writer:
#             tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#             tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
#         torch.cuda.empty_cache()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置后端为Agg以避免显示问题
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# 导入save_images函数
from utils.other_util import save_images
import os
from torch.utils.tensorboard import SummaryWriter
import wandb


def plot_statistics(data_dict, save_path='statistics_plot.png'):
    """
    绘制字典中每个值的折线图，并标记最大值、最小值、均值和中位数

    Parameters:
    data_dict: dict, 包含要绘制的数据的字典，格式为 {'name1': array1, 'name2': array2, ...}
    save_path: str, 保存路径
    """
    # 检查字典是否为空
    if not data_dict:
        print("警告：输入字典为空！")
        return

    # 创建图形和子图
    num_plots = len(data_dict)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))

    # 如果只有一个子图，确保axes是数组形式
    if num_plots == 1:
        axes = [axes]

    # 为每个数据项创建图表
    for ax, (name, data) in zip(axes, data_dict.items()):
        # 确保数据是numpy数组
        data_array = np.array(data)

        # 计算统计量（新增中位数）
        stats = {
            'max': np.max(data_array),
            'min': np.min(data_array),
            'mean': np.mean(data_array),
            'median': np.median(data_array),  # 新增中位数计算
            'max_idx': np.argmax(data_array),
            'min_idx': np.argmin(data_array)
        }

        # 绘制折线图
        x = np.arange(len(data_array))
        ax.plot(x, data_array, '-', linewidth=1, alpha=0.7, label=name)

        # 标记最大值
        ax.plot(stats['max_idx'], stats['max'], 'ro',
                markersize=8, label=f'Max: {stats["max"]:.4f}')
        ax.annotate(f'Max: {stats["max"]:.4f}',
                    xy=(stats['max_idx'], stats['max']),
                    xytext=(10, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))

        # 标记最小值
        ax.plot(stats['min_idx'], stats['min'], 'go',
                markersize=8, label=f'Min: {stats["min"]:.4f}')
        ax.annotate(f'Min: {stats["min"]:.4f}',
                    xy=(stats['min_idx'], stats['min']),
                    xytext=(10, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))

        # 添加均值线（橙色虚线）
        ax.axhline(y=stats['mean'], color='orange', linestyle='--',
                   linewidth=2, label=f'Mean: {stats["mean"]:.4f}')
        ax.text(len(data_array) * 0.8, stats['mean'] * 1.05,
                f'Mean: {stats["mean"]:.4f}',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))

        # 新增：添加中位数线（蓝色点划线）
        ax.axhline(y=stats['median'], color='blue', linestyle='-.',
                   linewidth=2, label=f'Median: {stats["median"]:.4f}')
        ax.text(len(data_array) * 0.6, stats['median'] * 1.05,
                f'Median: {stats["median"]:.4f}',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))

        ax.set_title(f'{name} Distribution with Statistics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 打印统计信息（新增中位数输出）
        print(f"{name} - 最大值: {stats['max']:.6f}, 最小值: {stats['min']:.6f}, "
              f"均值: {stats['mean']:.6f}, 中位数: {stats['median']:.6f}")

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"图表已保存至: {save_path}")


# def plot_statistics_with_annotations(max_scales, opacities, save_path='gaussian_statistics.png'):
#     """
#     绘制max_scales和opacities的折线图，并标记最大值、最小值和均值
#
#     Parameters:
#     max_scales: numpy array, 最大尺度值
#     opacities: numpy array, 不透明度值
#     save_path: str, 保存路径
#     """
#
#     # 创建图形和子图
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#
#     # 计算统计量
#     max_scales_stats = {
#         'max': np.max(max_scales),
#         'min': np.min(max_scales),
#         'mean': np.mean(max_scales),
#         'max_idx': np.argmax(max_scales),
#         'min_idx': np.argmin(max_scales)
#     }
#
#     opacities_stats = {
#         'max': np.max(opacities),
#         'min': np.min(opacities),
#         'mean': np.mean(opacities),
#         'max_idx': np.argmax(opacities),
#         'min_idx': np.argmin(opacities)
#     }
#
#     # 绘制max_scales折线图
#     x = np.arange(len(max_scales))
#     ax1.plot(x, max_scales, 'b-', linewidth=1, alpha=0.7, label='Max Scales')
#
#     # 标记最大值
#     ax1.plot(max_scales_stats['max_idx'], max_scales_stats['max'], 'ro',
#              markersize=8, label=f'Max: {max_scales_stats["max"]:.4f}')
#     ax1.annotate(f'Max: {max_scales_stats["max"]:.4f}',
#                  xy=(max_scales_stats['max_idx'], max_scales_stats['max']),
#                  xytext=(10, 10), textcoords='offset points',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
#                  bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))
#
#     # 标记最小值
#     ax1.plot(max_scales_stats['min_idx'], max_scales_stats['min'], 'go',
#              markersize=8, label=f'Min: {max_scales_stats["min"]:.4f}')
#     ax1.annotate(f'Min: {max_scales_stats["min"]:.4f}',
#                  xy=(max_scales_stats['min_idx'], max_scales_stats['min']),
#                  xytext=(10, -20), textcoords='offset points',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
#                  bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
#
#     # 添加均值线
#     ax1.axhline(y=max_scales_stats['mean'], color='orange', linestyle='--',
#                 linewidth=2, label=f'Mean: {max_scales_stats["mean"]:.4f}')
#     ax1.text(len(max_scales) * 0.8, max_scales_stats['mean'] * 1.05,
#              f'Mean: {max_scales_stats["mean"]:.4f}',
#              bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))
#
#     ax1.set_title('Max Scales Distribution with Statistics', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('Gaussian Index', fontsize=12)
#     ax1.set_ylabel('Max Scale Value', fontsize=12)
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
#
#     # 绘制opacities折线图
#     ax2.plot(x, opacities, 'purple', linewidth=1, alpha=0.7, label='Opacities')
#
#     # 标记最大值
#     ax2.plot(opacities_stats['max_idx'], opacities_stats['max'], 'ro',
#              markersize=8, label=f'Max: {opacities_stats["max"]:.4f}')
#     ax2.annotate(f'Max: {opacities_stats["max"]:.4f}',
#                  xy=(opacities_stats['max_idx'], opacities_stats['max']),
#                  xytext=(10, 10), textcoords='offset points',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
#                  bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))
#
#     # 标记最小值
#     ax2.plot(opacities_stats['min_idx'], opacities_stats['min'], 'go',
#              markersize=8, label=f'Min: {opacities_stats["min"]:.4f}')
#     ax2.annotate(f'Min: {opacities_stats["min"]:.4f}',
#                  xy=(opacities_stats['min_idx'], opacities_stats['min']),
#                  xytext=(10, -20), textcoords='offset points',
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
#                  bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
#
#     # 添加均值线
#     ax2.axhline(y=opacities_stats['mean'], color='orange', linestyle='--',
#                 linewidth=2, label=f'Mean: {opacities_stats["mean"]:.4f}')
#     ax2.text(len(opacities) * 0.8, opacities_stats['mean'] * 1.05,
#              f'Mean: {opacities_stats["mean"]:.4f}',
#              bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))
#
#     ax2.set_title('Opacities Distribution with Statistics', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('Gaussian Index', fontsize=12)
#     ax2.set_ylabel('Opacity Value', fontsize=12)
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#
#     # 调整布局
#     plt.tight_layout()
#
#     # 保存图像
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"图表已保存至: {save_path}")
#     print(
#         f"Max Scales - 最大值: {max_scales_stats['max']:.6f}, 最小值: {max_scales_stats['min']:.6f}, 均值: {max_scales_stats['mean']:.6f}")
#     print(
#         f"Opacities - 最大值: {opacities_stats['max']:.6f}, 最小值: {opacities_stats['min']:.6f}, 均值: {opacities_stats['mean']:.6f}")

# 使用示例
# 假设您已经有了 max_scales 和 opacities 这两个numpy数组
# max_scales = torch.max(gaussians.get_scaling, dim=1).values.detach().cpu().numpy()
# opacities = gaussians.get_opacity.cpu().numpy()

# 调用函数绘制图表
# plot_statistics_with_annotations(max_scales, opacities, 'gaussian_analysis.png')


def create_log_dir(save_path, args, config_path):
    """
    创建日志目录并返回路径
    
    Args:
        save_path: 基础保存路径
        args: 参数对象
        config_path: 配置文件路径
    
    Returns:
        tuple: (save_path, log_path, result_path)
    """
    # 创建保存路径
    full_save_path = os.path.join(save_path, f"{args.mode}")
    os.makedirs(full_save_path, exist_ok=True)
    
    # 创建日志路径
    log_path = os.path.join(full_save_path, "log.txt")
    
    # 创建结果路径
    result_path = os.path.join(full_save_path, "result")
    os.makedirs(result_path, exist_ok=True)
    
    return full_save_path, log_path, result_path


def initialize_logging(tps, args, save_path):
    """
    初始化日志系统（tensorboard, wandb等）
    
    Args:
        tps: 训练参数对象
        args: 参数对象
        save_path: 保存路径
    """
    # 初始化tensorboard writer
    if getattr(tps, 'use_tensorboard', False):
        tps.tb_writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard'))
    else:
        tps.tb_writer = None
    
    # 初始化wandb（如果可用）
    if getattr(tps, 'use_wandb', False) and wandb is not None:
        try:
            wandb.init(
                project="PhyAvatar",
                name=args.mode,
                config=vars(args),
                dir=save_path
            )
            tps.wandb_run = wandb.run
        except Exception as e:
            print(f"Wandb初始化失败: {e}")
            tps.wandb_run = None
    else:
        tps.wandb_run = None


def cleanup_logging(tps):
    """
    清理日志系统
    
    Args:
        tps: 训练参数对象
    """
    if tps.tb_writer is not None:
        tps.tb_writer.close()
    
    if tps.wandb_run is not None:
        wandb.finish()


def log_training_metrics(tps, iteration, test_loss, L1, ellipsoid_loss, lpips_loss, l1_mask_loss, ssim, psnr_val, opacity_loss, scale_loss, mode):
    """
    记录训练指标到各种日志系统
    
    Args:
        tps: 训练参数对象
        iteration: 当前迭代次数
        test_loss: 测试损失
        L1: L1损失
        ellipsoid_loss: 椭球损失
        lpips_loss: LPIPS损失
        l1_mask_loss: L1掩码损失
        ssim: SSIM值
        psnr_val: PSNR值
        opacity_loss: 不透明度损失
        scale_loss: 尺度损失
        mode: 模式名称
    """
    def to_scalar(x):
        try:
            import torch
            import numpy as np
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.item()
                return x.detach().mean().item()
            if isinstance(x, np.ndarray):
                return float(x.mean())
            return float(x)
        except Exception:
            # 兜底：直接返回原值（若TB/W&B不能接受，会在上层暴露）
            return x

    # TensorBoard日志
    if tps.tb_writer is not None:
        tps.tb_writer.add_scalar("test_loss/total_loss", to_scalar(test_loss), iteration)
        tps.tb_writer.add_scalar("test_loss/L1", to_scalar(L1), iteration)
        tps.tb_writer.add_scalar("test_loss/ellipsoid_loss", to_scalar(ellipsoid_loss), iteration)
        tps.tb_writer.add_scalar("test_loss/lpips_loss", to_scalar(lpips_loss), iteration)
        tps.tb_writer.add_scalar("test_loss/l1_mask_loss", to_scalar(l1_mask_loss), iteration)
        tps.tb_writer.add_scalar("test_loss/ssim", to_scalar(ssim), iteration)
        tps.tb_writer.add_scalar("test_loss/psnr", to_scalar(psnr_val), iteration)
        tps.tb_writer.add_scalar("test_loss/opacity_loss", to_scalar(opacity_loss), iteration)
        tps.tb_writer.add_scalar("test_loss/scale_loss", to_scalar(scale_loss), iteration)
    
    # Wandb日志
    if tps.wandb_run is not None:
        wandb.log({
            "iteration": iteration,
            "test_loss": to_scalar(test_loss),
            "L1": to_scalar(L1),
            "ellipsoid_loss": to_scalar(ellipsoid_loss),
            "lpips_loss": to_scalar(lpips_loss),
            "l1_mask_loss": to_scalar(l1_mask_loss),
            "ssim": to_scalar(ssim),
            "psnr": to_scalar(psnr_val),
            "opacity_loss": to_scalar(opacity_loss),
            "scale_loss": to_scalar(scale_loss),
            "mode": mode
        })


def append_to_log(log_path, message):
    """
    追加消息到日志文件
    
    Args:
        log_path: 日志文件路径
        message: 要记录的消息
    """
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")
def create_mask_overlay(gt_mask: np.ndarray, render_alpha: np.ndarray) -> np.ndarray:
    """
    Creates a color overlay to visualize the difference between two masks.

    It assigns the ground truth mask to the red channel and the rendered alpha
    mask to the blue channel of a new RGB image.

    Args:
        gt_mask (np.ndarray): The ground truth mask, expected to be a 2D array
                              with float values in the range [0, 1].
        render_alpha (np.ndarray): The rendered alpha mask, expected to be a 2D
                                   array with the same shape as gt_mask and
                                   float values in the range [0, 1].

    Returns:
        np.ndarray: An RGB image (H, W, 3) as a uint8 NumPy array where:
                    - Red channel corresponds to gt_mask.
                    - Blue channel corresponds to render_alpha.
                    - Magenta indicates overlap.
    """

    
    # --- Input Validation ---
    if gt_mask.shape != render_alpha.shape:
        raise ValueError("Input masks must have the same shape.")
    if gt_mask.ndim != 2:
        raise ValueError("Input masks must be 2D single-channel arrays.")

    h, w = gt_mask.shape

    # --- Create Canvas ---
    # Create a black RGB canvas of type uint8 (0-255)
    overlay_image = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Normalize and Convert Masks ---
    # Convert the float masks [0, 1] to uint8 masks [0, 255]
    gt_mask_u8 = (gt_mask * 255).astype(np.uint8)
    render_alpha_u8 = (render_alpha * 255).astype(np.uint8)

    # --- Assign Channels ---
    # Assign gt_mask to the Red channel (index 0)
    overlay_image[:, :, 0] = gt_mask_u8
    
    # Assign render_alpha to the Blue channel (index 2)
    overlay_image[:, :, 2] = render_alpha_u8

    return overlay_image