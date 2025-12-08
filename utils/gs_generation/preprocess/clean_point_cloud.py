#!/usr/bin/env python3
"""
使用 mask 过滤点云，采用投票机制：每个点如果在多个视角的 mask 区域内，则保留。

与 filter_point_cloud_with_masks.py 的区别：
- 本脚本使用投票机制：点如果在 threshold 比例的视角中落在 mask 内，则保留
- filter_point_cloud_with_masks.py 使用交集机制：点必须在所有视角的 mask 内才保留

Usage:
    python utils/gs_generation/preprocess/clean_point_cloud.py \
        --ply path/to/sparse_pc.ply \
        --json path/to/transforms.json \
        --output path/to/cleaned_pc.ply \
        --thresh 0.3
"""

import json
import numpy as np
import cv2
import os
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import argparse


def get_camera_matrix(frame):
    """从 transforms.json 的 frame 中构建内参和外参矩阵"""
    # 1. 获取内参 K
    fl_x = frame.get('fl_x')
    fl_y = frame.get('fl_y', fl_x)
    cx = frame.get('cx')
    cy = frame.get('cy')
    w = frame.get('w')
    h = frame.get('h')
    
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    
    # 2. 获取外参 World-to-Camera (c2w 的逆)
    # Nerfstudio transforms.json 存储的是 c2w (Camera to World)
    # 与 filter_point_cloud_with_masks.py 保持一致，使用 float64 精度
    c2w = np.array(frame['transform_matrix'], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    return K, w2c, w, h


def project_points(points, K, w2c, w=None, h=None, flip_x=False, flip_y=False):
    """将 3D 点投影到 2D 像素坐标
    
    注意：Nerfstudio 使用 OpenGL 坐标系，相机朝向 -Z 方向
    点在相机前方时，z < 0
    
    此实现与 filter_point_cloud_with_masks.py 保持一致
    """
    # 1. 转为齐次坐标 (N, 4)
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    
    # 2. 变换到相机坐标系
    # w2c 是 (4, 4) 矩阵，points_homogeneous 是 (N, 4)
    # 计算: w2c @ points_homogeneous.T -> (4, N)，然后转置得到 (N, 4)
    points_cam = (w2c @ points_homogeneous.T).T[:, :3]
    
    # 3. 获取深度（Nerfstudio: z < 0 表示在相机前方）
    z = points_cam[:, 2]
    depths = z
    
    # 4. 投影到像素平面
    # Nerfstudio 坐标系：相机朝向 -Z，所以投影时使用 -z
    # 与 filter_point_cloud_with_masks.py 完全一致
    x_norm = points_cam[:, 0] / (-z + 1e-8)
    y_norm = points_cam[:, 1] / (-z + 1e-8)
    
    # 注意：OpenGL 中相机 Y 轴向上，但图像 v 轴向下
    # 如果投影结果上下翻转，可能需要对 y_norm 取反
    # 但 filter_point_cloud_with_masks.py 没有取反，所以这里也不取反
    
    # 使用内参矩阵投影
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    u = fx * x_norm + cx
    v = fy * y_norm + cy
    
    # 可选的坐标翻转（用于调试坐标系问题）
    # 如果投影结果有一半缺失，可能需要翻转某个轴
    if flip_x and w is not None:
        u = w - 1 - u
    if flip_y and h is not None:
        v = h - 1 - v
    
    uv = np.stack([u, v], axis=1)
    
    return uv, depths


def filter_point_cloud(ply_path, json_path, output_path, threshold=0.1, flip_y=False, flip_x=False, erode_size=0, verbose_mask_check=False):
    print(f"Loading point cloud from {ply_path}...")
    plydata = PlyData.read(ply_path)
    
    # 提取顶点数据
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # 初始化投票箱：每个点被 mask 判定为"保留"的次数
    keep_votes = np.zeros(points.shape[0], dtype=np.int32)
    # 记录每个点被多少个相机"看到"（在视锥体内）
    seen_counts = np.zeros(points.shape[0], dtype=np.int32)
    
    print(f"Loading transforms from {json_path}...")
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    # 遍历所有相机帧
    frames = meta['frames']
    base_dir = os.path.dirname(json_path)
    
    print(f"Total frames: {len(frames)}")
    print("Projecting points to masks...")
    processed_frames = 0
    for frame in tqdm(frames):
        # 1. 读取 Mask
        mask_path = frame.get('mask_path') # 确保你的 json 里有 mask_path
        if not mask_path:
            # 如果 json 没有 mask_path，尝试用 file_path 替换后缀推断
            # 这里的逻辑需要根据你的实际文件结构修改
            # 假设 mask 在 masks 文件夹下，且文件名相同
            img_path = frame['file_path']
            # 示例：images/001.png -> masks/001.png
            mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
            
        full_mask_path = os.path.join(base_dir, mask_path)
        if not os.path.exists(full_mask_path):
            continue
        
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        processed_frames += 1
            
        # 二值化 Mask (确保白色是 255，黑色是 0)
        # 使用阈值 127，小于 127 的设为 0，大于等于 127 的设为 255
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 确保 mask 是纯黑白的（只有 0 和 255，没有灰色）
        # 检查是否有中间值（灰色）
        unique_values = np.unique(mask)
        non_binary_values = unique_values[(unique_values != 0) & (unique_values != 255)]
        if len(non_binary_values) > 0:
            print(f"  ⚠️  Warning: Mask {os.path.basename(full_mask_path)} contains non-binary values: {non_binary_values}")
            print(f"    Re-binarizing mask...")
            # 强制二值化：所有非 0 的值都设为 255
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        # 详细检查 mask 统计信息（可选）
        if verbose_mask_check and processed_frames % 10 == 0:
            unique_after = np.unique(mask)
            white_pixels = np.sum(mask == 255)
            black_pixels = np.sum(mask == 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            white_ratio = white_pixels / total_pixels * 100
            print(f"  Frame {processed_frames}: Mask stats - unique values: {unique_after}, "
                  f"white: {white_pixels} ({white_ratio:.1f}%), black: {black_pixels}")
        
        # 形态学操作：腐蚀 mask 以去除边界白色线
        # erode_size > 0 时会先腐蚀 mask，去除边界附近的像素
        if erode_size > 0:
            # 方法1：使用距离变换去除边界（更精确）
            # 计算到边界的距离，只保留距离边界足够远的像素
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            # 使用更大的阈值来更彻底地去除边界（erode_size * 1.5）
            dist_threshold = max(erode_size * 1.5, erode_size + 2)
            mask = np.where(dist_transform > dist_threshold, 255, 0).astype(np.uint8)
            
            # 方法2：多次腐蚀以更彻底地去除边界
            # 使用椭圆核（比方形核更平滑）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            # 进行多次腐蚀迭代
            mask = cv2.erode(mask, kernel, iterations=2)
            
            # 方法3：明确检测并去除边界区域
            # 找到所有边界像素（与黑色相邻的白色像素）
            # 先膨胀再减去原图，得到边界
            kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_dilated = cv2.dilate(mask, kernel_boundary, iterations=1)
            boundary = mask_dilated - mask
            # 从 mask 中减去边界
            mask = cv2.subtract(mask, boundary)
            
            # 最终确保是纯黑白的
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        # 2. 获取投影矩阵
        K, w2c, w, h = get_camera_matrix(frame)
        
        # 3. 投影所有点
        uv, depths = project_points(points, K, w2c, w=w, h=h, flip_x=flip_x, flip_y=flip_y)
        
        # 4. 筛选：必须在相机前方 (z < 0 for Nerfstudio) 且在图像范围内
        # Nerfstudio 使用 OpenGL 坐标系，相机朝向 -Z，所以 z < 0 表示在相机前方
        in_front = depths < 0
        in_bounds_u = (uv[:, 0] >= 0) & (uv[:, 0] < w)
        in_bounds_v = (uv[:, 1] >= 0) & (uv[:, 1] < h)
        valid_proj = in_front & in_bounds_u & in_bounds_v
        
        # 记录被看到的次数
        seen_counts[valid_proj] += 1
        
        # 5. 检查 Mask 值
        # 注意：numpy 数组索引是 [row, col] = [height, width] = [v, u]
        # 所以 mask[v, u] 是正确的索引方式
        u_coords = uv[valid_proj, 0].astype(int)
        v_coords = uv[valid_proj, 1].astype(int)
        
        # 确保坐标在有效范围内
        u_coords = np.clip(u_coords, 0, w - 1)
        v_coords = np.clip(v_coords, 0, h - 1)
        
        # 调试：检查投影范围（每10帧打印一次）
        if processed_frames % 10 == 0 and np.sum(valid_proj) > 0:
            u_min, u_max = np.min(u_coords), np.max(u_coords)
            v_min, v_max = np.min(v_coords), np.max(v_coords)
            u_center = (u_min + u_max) / 2
            v_center = (v_min + v_max) / 2
            print(f"  Frame {processed_frames}: u range [{u_min}, {u_max}] (center: {u_center:.1f}) / {w}, "
                  f"v range [{v_min}, {v_max}] (center: {v_center:.1f}) / {h}")
            # 如果投影范围明显偏向一边，可能是坐标系问题
            if u_center < w * 0.3 or u_center > w * 0.7:
                print(f"    ⚠️  Warning: u center ({u_center:.1f}) is far from image center ({w/2:.1f})")
            if v_center < h * 0.3 or v_center > h * 0.7:
                print(f"    ⚠️  Warning: v center ({v_center:.1f}) is far from image center ({h/2:.1f})")
        
        # 获取对应像素的 Mask 值
        # numpy 数组索引：mask[row, col] = mask[v, u]
        mask_values = mask[v_coords, u_coords]
        
        # 确保 mask_values 是纯黑白的（调试用）
        # 如果 mask 已经正确二值化，这里应该只有 0 和 255
        unique_mask_vals = np.unique(mask_values)
        if len(unique_mask_vals) > 2 or (len(unique_mask_vals) == 2 and not (0 in unique_mask_vals and 255 in unique_mask_vals)):
            if processed_frames % 10 == 0:  # 每10帧打印一次警告
                print(f"  Warning: Mask values are not binary: {unique_mask_vals}")
        
        # 如果 Mask 是白色的 (255)，则投一票"保留"
        # 使用严格比较：只有完全等于 255 的才算白色
        # 获取 valid_proj 为 True 的点的索引
        valid_indices = np.where(valid_proj)[0]
        # 只给落在白色区域的点加分（严格检查 == 255）
        in_mask = mask_values == 255
        in_mask_count = np.sum(in_mask)
        keep_votes[valid_indices[in_mask]] += 1
        
        # 调试信息（每10帧打印一次）
        if processed_frames % 10 == 0:
            mask_coverage = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1]) * 100
            # 检查深度范围，帮助诊断坐标系问题
            if np.sum(valid_proj) > 0:
                valid_depths = depths[valid_proj]
                print(f"  Frame {processed_frames}: {np.sum(valid_proj)} points visible, {in_mask_count} in mask, "
                      f"mask coverage: {mask_coverage:.1f}%, "
                      f"depth range: [{np.min(valid_depths):.2f}, {np.max(valid_depths):.2f}]")
            else:
                print(f"  Frame {processed_frames}: {np.sum(valid_proj)} points visible, mask coverage: {mask_coverage:.1f}%")
    
    # === 最终决策 ===
    # 如果一个点在它出现的所有视野中，有超过 threshold 比例的情况落在白色 Mask 里，则保留
    # 避免除零错误
    ratios = np.zeros_like(keep_votes, dtype=np.float32)
    valid_seen = seen_counts > 0
    ratios[valid_seen] = keep_votes[valid_seen].astype(np.float32) / seen_counts[valid_seen]
    
    # 打印一些统计信息，帮助诊断
    if np.sum(valid_seen) > 0:
        print(f"\n=== Voting Statistics ===")
        print(f"Points visible in at least one view: {np.sum(valid_seen)}")
        print(f"Average votes per visible point: {np.mean(keep_votes[valid_seen]):.2f}")
        print(f"Average seen count per visible point: {np.mean(seen_counts[valid_seen]):.2f}")
        print(f"Average ratio (votes/seen) per visible point: {np.mean(ratios[valid_seen]):.3f}")
        print(f"Points with ratio > 0.1: {np.sum(ratios[valid_seen] > 0.1)}")
        print(f"Points with ratio > 0.2: {np.sum(ratios[valid_seen] > 0.2)}")
        print(f"Points with ratio > 0.3: {np.sum(ratios[valid_seen] > 0.3)}")
        print(f"Points with ratio > 0.5: {np.sum(ratios[valid_seen] > 0.5)}")
    
    # 筛选点
    # threshold = 0.1 意味着：只要有 10% 的相机认为它是四肢，就保留（比较宽松）
    # 如果你想切得更干净，可以设为 0.5 或更高
    mask_keep = ratios > threshold
    
    print(f"\n=== Filtering Summary ===")
    print(f"Processed frames: {processed_frames}/{len(frames)}")
    print(f"Original points: {points.shape[0]}")
    print(f"Points seen by at least one camera: {np.sum(seen_counts > 0)}")
    print(f"Filtered points: {np.sum(mask_keep)}")
    print(f"Removed points: {points.shape[0] - np.sum(mask_keep)}")
    print(f"Removal ratio: {(points.shape[0] - np.sum(mask_keep)) / points.shape[0] * 100:.2f}%")
    print(f"Threshold used: {threshold}")
    
    # 打印一些统计信息
    if np.sum(seen_counts > 0) == 0:
        print("\n⚠️  WARNING: No points were visible in any camera!")
        print("   This might indicate a coordinate system mismatch.")
        print("   Check if camera parameters and point cloud coordinate system match.")
    elif np.sum(mask_keep) == 0:
        print("\n⚠️  WARNING: All visible points were filtered out!")
        print(f"   Points visible: {np.sum(seen_counts > 0)}")
        if np.sum(seen_counts > 0) > 0:
            print(f"   Average votes per visible point: {np.mean(keep_votes[seen_counts > 0]):.2f}")
            print(f"   Average seen count per visible point: {np.mean(seen_counts[seen_counts > 0]):.2f}")
            print(f"   Average ratio (votes/seen): {np.mean(ratios[seen_counts > 0]):.3f}")
        print(f"   Try lowering --thresh (current: {threshold})")
    else:
        # 如果过滤后有点，打印一些有用的信息
        kept_ratios = ratios[mask_keep]
        if len(kept_ratios) > 0:
            print(f"\nKept points statistics:")
            print(f"   Min ratio: {np.min(kept_ratios):.3f}")
            print(f"   Max ratio: {np.max(kept_ratios):.3f}")
            print(f"   Mean ratio: {np.mean(kept_ratios):.3f}")
            print(f"   Median ratio: {np.median(kept_ratios):.3f}")
    
    # 保存新的 PLY
    # 只需要保留 'vertex' 元素中对应的点
    new_vertices = vertices[mask_keep]
    new_ply = PlyData([PlyElement.describe(new_vertices, 'vertex')], text=False)
    new_ply.write(output_path)
    print(f"Saved filtered point cloud to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter point cloud using voting mechanism based on masks"
    )
    parser.add_argument("--ply", required=True, help="Original sparse point cloud path (.ply)")
    parser.add_argument("--json", required=True, help="transforms.json path")
    parser.add_argument("--output", required=True, help="Output path for cleaned .ply")
    parser.add_argument("--thresh", type=float, default=0.2, help="Keep threshold (0.0 - 1.0), default 0.2")
    parser.add_argument("--flip-x", action="store_true", help="Flip X (u) coordinate (for debugging coordinate system)")
    parser.add_argument("--flip-y", action="store_true", help="Flip Y (v) coordinate (for debugging coordinate system)")
    parser.add_argument("--erode-size", type=int, default=7, 
                        help="Erosion kernel size to remove white lines at mask boundaries (0 to disable, default 7)")
    parser.add_argument("--verbose-mask-check", action="store_true",
                        help="Print detailed mask statistics for debugging")
    args = parser.parse_args()
    
    filter_point_cloud(args.ply, args.json, args.output, args.thresh, 
                      flip_x=args.flip_x, flip_y=args.flip_y, erode_size=args.erode_size,
                      verbose_mask_check=args.verbose_mask_check)

