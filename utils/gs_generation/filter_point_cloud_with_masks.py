#!/usr/bin/env python3
"""
使用 mask 过滤点云，只保留在 mask 区域内的点。

Usage:
    python filter_point_cloud_with_masks.py \
        --ply_path /path/to/sparse_pcd.ply \
        --transforms_json /path/to/transforms.json \
        --output_ply_path /path/to/filtered_sparse_pcd.ply \
        [--mask_threshold 0.5] \
        [--min_views 1]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import trimesh

# 确保可以从项目根目录导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_frame_by_view_index(frames, view_index: int):
    """通过 camera_label 或 mask_path 找到对应的 frame（与 project_mask_to_3d.py 保持一致）"""
    view_label = f"{view_index:02d}"
    
    for frame_idx, frame in enumerate(frames):
        camera_label = frame.get("camera_label", f"{frame_idx:02d}")
        
        # 方法1: 通过 camera_label 匹配
        if camera_label == view_label or str(camera_label) == str(view_index):
            return frame, frame_idx
        
        # 方法2: 从 mask_path 中提取视角编号
        mask_path = frame.get("mask_path", "")
        if mask_path:
            import re
            match = re.search(r'[/\\](\d+)[/\\]', str(mask_path))
            if match and int(match.group(1)) == view_index:
                return frame, frame_idx
    
    return None, None


def filter_point_cloud_with_masks(
    ply_path: Path,
    transforms_json_path: Path,
    output_ply_path: Path,
    mask_threshold: float = 0.5,
    min_views: int = 1,
    view_indices: Optional[List[int]] = None,
) -> bool:
    """
    使用 mask 过滤点云，只保留在 mask 区域内的点。
    
    Args:
        ply_path: 输入点云文件路径
        transforms_json_path: transforms.json 文件路径（包含相机参数和 mask_path）
        output_ply_path: 输出点云文件路径
        mask_threshold: mask 阈值（0-1），默认 0.5
        min_views: 点需要在至少多少个视角的 mask 内才保留，默认 1
    
    Returns:
        bool: 是否成功
    """
    try:
        # 1. 读取点云
        if not ply_path.exists():
            print(f"[WARN] Point cloud not found: {ply_path}")
            return False
        
        mesh = trimesh.load(str(ply_path))
        if not hasattr(mesh, 'vertices'):
            print(f"[WARN] Failed to load point cloud from {ply_path}")
            return False
        
        points = mesh.vertices  # (N, 3)
        print(f"[INFO] Loaded {len(points)} points from {ply_path}")
        
        # 2. 读取相机参数
        if not transforms_json_path.exists():
            print(f"[WARN] Transforms.json not found: {transforms_json_path}")
            return False
        
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)
        
        frames = transforms_data.get("frames", [])
        if not frames:
            print(f"[WARN] No frames found in {transforms_json_path}")
            return False
        
        # 3. 依次对每个视角用 mask 过滤点云（顺序叠加）
        # 维护当前剩余点的索引
        remaining_indices = np.arange(len(points))
        processed_views = 0
        
        # 如果指定了视角索引，使用与 project_mask_to_3d.py 相同的匹配逻辑
        if view_indices is not None:
            frames_to_process = []
            frame_indices_map = {}  # 映射：frame_idx -> original_frame_idx
            
            for view_idx in view_indices:
                frame, frame_idx = find_frame_by_view_index(frames, view_idx)
                if frame is not None:
                    frames_to_process.append(frame)
                    frame_indices_map[len(frames_to_process) - 1] = frame_idx
                    camera_label = frame.get("camera_label", f"{frame_idx:02d}")
                    mask_path = frame.get("mask_path", "")
                    print(f"[INFO] Matched view {view_idx}: frame_idx={frame_idx}, camera_label={camera_label}, mask_path={mask_path}")
                else:
                    print(f"[WARN] View {view_idx} not found in transforms.json")
            
            print(f"[INFO] Processing only views: {view_indices} (matched {len(frames_to_process)} frames)")
            if len(frames_to_process) == 0:
                print(f"[ERROR] No frames found matching views {view_indices}")
                print(f"[DEBUG] Available camera_labels: {[f.get('camera_label', f'{i:02d}') for i, f in enumerate(frames[:10])]}...")
                print(f"[DEBUG] Sample mask_paths: {[f.get('mask_path', '') for f in frames[:5]]}...")
                return False
        else:
            frames_to_process = frames
            frame_indices_map = {i: i for i in range(len(frames))}
        
        for frame_idx, frame in enumerate(frames_to_process):
            # 获取原始视角索引（用于调试信息）
            original_view_idx = frame_indices_map.get(frame_idx, frame_idx)
            camera_label = frame.get("camera_label", f"{original_view_idx:02d}")
            if len(remaining_indices) == 0:
                print(f"[WARN] No points remaining, stopping at view {processed_views}")
                break
            
            # 获取当前剩余的点
            current_points = points[remaining_indices]
            
            # 获取相机参数
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            
            fx = float(frame.get("fl_x", frame.get("fx", 0)))
            fy = float(frame.get("fl_y", frame.get("fy", 0)))
            cx = float(frame.get("cx", 0))
            cy = float(frame.get("cy", 0))
            w = int(frame.get("w", 0))
            h = int(frame.get("h", 0))
            
            if fx == 0 or fy == 0 or w == 0 or h == 0:
                continue
            
            # 获取 mask 路径
            mask_path_str = frame.get("mask_path", "")
            if not mask_path_str:
                print(f"[DEBUG] Frame {original_view_idx} (processed {processed_views}) has no mask_path, skipping")
                continue
            
            # 构建完整的 mask 路径
            mask_path_str_clean = mask_path_str.lstrip("./")
            mask_path = transforms_json_path.parent / mask_path_str_clean
            
            if not mask_path.exists():
                print(f"[WARN] Mask not found: {mask_path} (resolved from {mask_path_str})")
                continue
            
            print(f"[DEBUG] Processing mask: {mask_path}")
            print(f"[DEBUG]   camera_label: {camera_label}, frame_idx: {original_view_idx}")
            print(f"[DEBUG]   mask_path from frame: {frame.get('mask_path', 'N/A')}")
            
            # 读取 mask
            try:
                mask_img = Image.open(mask_path).convert("L")
                mask_w, mask_h = mask_img.size
                
                if mask_h != h or mask_w != w:
                    print(f"[DEBUG] Resizing mask from ({mask_w}, {mask_h}) to ({w}, {h})")
                    mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)
                
                mask = np.array(mask_img, dtype=np.float32) / 255.0
                mask_bool = mask >= mask_threshold
                
                assert mask_bool.shape == (h, w), f"Mask shape {mask_bool.shape} != image shape ({h}, {w})"
                
                mask_coverage = np.sum(mask_bool) / mask_bool.size
                print(f"[DEBUG] Mask loaded: shape={mask_bool.shape}, coverage={mask_coverage*100:.1f}%")
            except Exception as e:
                print(f"[WARN] Failed to load mask {mask_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 投影当前剩余的点云到当前相机
            points_homogeneous = np.hstack([current_points, np.ones((len(current_points), 1))])
            points_cam = (w2c @ points_homogeneous.T).T[:, :3]
            
            # 检查点是否在相机前方
            z = points_cam[:, 2]
            in_front = z < 0  # nerfstudio convention
            
            # 投影到像素坐标
            x_norm = points_cam[:, 0] / (-z + 1e-8)
            y_norm = points_cam[:, 1] / (-z + 1e-8)
            u = fx * x_norm + cx
            v = fy * y_norm + cy
            
            # 检查是否在图像范围内
            in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            valid = in_front & in_bounds
            
            # 检查 mask：只保留在 mask 内的点
            # 对于不在当前视角可见的点，无法判断是否在 mask 内，所以删除
            keep_mask = np.zeros(len(current_points), dtype=bool)  # 默认删除所有点
            
            if np.any(valid):
                u_int = np.clip(np.round(u[valid]).astype(int), 0, w - 1)
                v_int = np.clip(np.round(v[valid]).astype(int), 0, h - 1)
                
                # 检查索引范围
                u_min, u_max = np.min(u_int), np.max(u_int)
                v_min, v_max = np.min(v_int), np.max(v_int)
                
                # 检查 mask 在这些位置的值
                # numpy array 索引是 [row, col] = [height, width] = [v, u]
                in_mask = mask_bool[v_int, u_int]
                
                # 对于在当前视角可见的点，只有在 mask 内的才保留
                # 对于不在当前视角可见的点，keep_mask 保持为 False（删除）
                keep_mask[valid] = in_mask
                
                print(f"[DEBUG]   Points visible: {np.sum(valid)}, in mask: {np.sum(in_mask)}, will keep: {np.sum(keep_mask)}")
                
                # 如果当前视角没有任何点在 mask 内，检查是否是投影问题
                if not np.any(in_mask):
                    # 检查 mask 在投影区域内的值
                    mask_region = mask_bool[v_min:v_max+1, u_min:u_max+1] if v_max >= v_min and u_max >= u_min else mask_bool
                    mask_region_white = np.sum(mask_region) if mask_region.size > 0 else 0
                    
                    print(f"[WARN] View {camera_label} (frame_idx: {original_view_idx}, processed {processed_views}): No points in mask (but {np.sum(valid)} points visible), skipping this view")
                    print(f"[DEBUG]   Projected u range: [{u_min}, {u_max}] (image width: {w})")
                    print(f"[DEBUG]   Projected v range: [{v_min}, {v_max}] (image height: {h})")
                    print(f"[DEBUG]   Mask region [{v_min}:{v_max+1}, {u_min}:{u_max+1}] has {mask_region_white} white pixels")
                    print(f"[DEBUG]   Full mask has {np.sum(mask_bool)} white pixels out of {mask_bool.size}")
                    processed_views += 1
                    continue
            else:
                # 如果当前视角没有任何点可见，跳过这个视角
                print(f"[WARN] View {camera_label} (frame_idx: {original_view_idx}, processed {processed_views}): No points visible, skipping this view")
                processed_views += 1
                continue
            
            # 更新剩余点的索引
            remaining_indices = remaining_indices[keep_mask]
            
            # 统计信息
            num_before = len(current_points)
            num_after = len(remaining_indices)
            num_removed = num_before - num_after
            num_valid = np.sum(valid)
            num_in_mask = np.sum(keep_mask[valid]) if np.any(valid) else 0
            
            print(f"[DEBUG] View {camera_label} (frame_idx: {original_view_idx}, processed {processed_views}): {num_in_mask}/{num_valid} valid points in mask, "
                  f"{num_removed} points removed, {num_after}/{num_before} points remaining")
            
            processed_views += 1
        
        print(f"[INFO] Processed {processed_views} views with masks")
        print(f"[DEBUG] Final points remaining: {len(remaining_indices)}/{len(points)} "
              f"({len(remaining_indices)/len(points)*100:.1f}%)")
        
        # 4. 获取过滤后的点云
        if len(remaining_indices) == 0:
            print(f"[WARN] No points remain after filtering, using original point cloud")
            filtered_points = points
        else:
            filtered_points = points[remaining_indices]
        
        print(f"[INFO] Filtered point cloud: {len(filtered_points)}/{len(points)} points remain "
              f"({len(filtered_points)/len(points)*100:.1f}%)")
        
        if len(filtered_points) == len(points):
            print(f"[WARN] No points were filtered! All points passed the mask check.")
            print(f"[WARN] This might indicate a problem with mask loading or projection.")
        
        if len(filtered_points) == 0:
            print(f"[WARN] No points remain after filtering, using original point cloud")
            filtered_points = points
        
        # 5. 保存过滤后的点云
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建新的 mesh 并保存
        filtered_mesh = trimesh.Trimesh(vertices=filtered_points)
        filtered_mesh.export(str(output_ply_path))
        
        print(f"[INFO] Saved filtered point cloud to {output_ply_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to filter point cloud: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter point cloud using masks from transforms.json"
    )
    parser.add_argument(
        "--ply_path",
        type=Path,
        required=True,
        help="Input point cloud PLY file path"
    )
    parser.add_argument(
        "--transforms_json",
        type=Path,
        required=True,
        help="Transforms.json file path (contains camera parameters and mask_path)"
    )
    parser.add_argument(
        "--output_ply_path",
        type=Path,
        required=True,
        help="Output filtered point cloud PLY file path"
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="Mask threshold (0-1), default 0.5"
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=1,
        help="Minimum number of views where point must be in mask, default 1"
    )
    parser.add_argument(
        "--view_indices",
        type=int,
        nargs="+",
        default=None,
        help="Only process specific view indices (e.g., --view_indices 48 49 50 51). If not specified, process all views."
    )
    
    args = parser.parse_args()
    
    success = filter_point_cloud_with_masks(
        ply_path=args.ply_path,
        transforms_json_path=args.transforms_json,
        output_ply_path=args.output_ply_path,
        mask_threshold=args.mask_threshold,
        min_views=args.min_views,
        view_indices=args.view_indices,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

