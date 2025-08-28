import sys
sys.path.append("gaussian-splatting")

import torch

def filter_large_scales(gaussians, visibility_filter, scale_threshold=0.005):
    """
    从visibility_filter中去除最大scale大于阈值的点

    Args:
        gaussians: 高斯模型实例
        visibility_filter: 原始的可见性掩码
        scale_threshold: 尺度阈值，默认0.1

    Returns:
        修改后的visibility_filter
    """
    # 获取所有高斯点的尺度
    scaling = gaussians.get_scaling  # shape: [N, 3]

    # 计算每个点的最大尺度
    max_scaling = torch.max(scaling, dim=1).values  # shape: [N]

    # 创建尺度过滤掩码：最大尺度 <= 阈值
    scale_mask = max_scaling <= scale_threshold
    # 组合可见性掩码和尺度掩码：既要可见又要尺度合格
    filtered_visibility = torch.logical_and(visibility_filter, scale_mask)

    return filtered_visibility

