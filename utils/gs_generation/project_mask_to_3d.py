#!/usr/bin/env python3
"""
将指定视角的相机位置和 mask 投影到 3D 空间，生成 PLY 文件。

Usage:
    python project_mask_to_3d.py \
        --transforms_json /path/to/transforms.json \
        --view_index 50 \
        --output_ply_path /path/to/mask_projection.ply \
        [--depth 5.0] \
        [--num_samples 10000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

# 确保可以从项目根目录导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_frame_by_view_index(frames, view_index: int):
    """通过 camera_label 或 mask_path 找到对应的 frame"""
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


def project_mask_to_3d(
    transforms_json_path: Path,
    view_index: int,
    output_ply_path: Path,
    near_depth: float = 1.0,
    far_depth: float = 10.0,
    num_depth_layers: int = 10,
    num_samples: int = 10000,
) -> bool:
    """
    将指定视角的相机位置和 mask 投影到 3D 空间，生成 PLY 文件。
    
    Args:
        transforms_json_path: transforms.json 文件路径
        view_index: 视角索引（如 50）
        output_ply_path: 输出 PLY 文件路径
        near_depth: 近平面深度（从相机到近平面的距离）
        far_depth: 远平面深度（从相机到远平面的距离）
        num_depth_layers: 深度层数（在近远平面之间生成多个深度层）
        num_samples: 每层 mask 区域的采样点数
    
    Returns:
        bool: 是否成功
    """
    try:
        # 1. 读取 transforms.json
        if not transforms_json_path.exists():
            print(f"[ERROR] Transforms.json not found: {transforms_json_path}")
            return False
        
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)
        
        frames = transforms_data.get("frames", [])
        if not frames:
            print(f"[ERROR] No frames found in {transforms_json_path}")
            return False
        
        # 2. 找到指定视角的 frame
        frame, frame_idx = find_frame_by_view_index(frames, view_index)
        if frame is None:
            print(f"[ERROR] View {view_index} not found in transforms.json")
            print(f"[DEBUG] Available camera_labels: {[f.get('camera_label', f'{i:02d}') for i, f in enumerate(frames[:10])]}...")
            return False
        
        camera_label = frame.get("camera_label", f"{frame_idx:02d}")
        print(f"[INFO] Found view {view_index} (camera_label: {camera_label}, frame_idx: {frame_idx})")
        
        # 3. 获取相机参数
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        
        fx = float(frame.get("fl_x", frame.get("fx", 0)))
        fy = float(frame.get("fl_y", frame.get("fy", 0)))
        cx = float(frame.get("cx", 0))
        cy = float(frame.get("cy", 0))
        w = int(frame.get("w", 0))
        h = int(frame.get("h", 0))
        
        if fx == 0 or fy == 0 or w == 0 or h == 0:
            print(f"[ERROR] Invalid camera parameters")
            return False
        
        # 4. 获取相机位置（c2w 的平移部分）
        camera_pos = c2w[:3, 3]
        print(f"[INFO] Camera position: {camera_pos}")
        
        # 5. 读取 mask
        mask_path_str = frame.get("mask_path", "")
        if not mask_path_str:
            print(f"[ERROR] No mask_path in frame")
            return False
        
        # 构建完整的 mask 路径
        mask_path_str_clean = mask_path_str.lstrip("./")
        mask_path = transforms_json_path.parent / mask_path_str_clean
        
        if not mask_path.exists():
            print(f"[ERROR] Mask not found: {mask_path}")
            return False
        
        print(f"[INFO] Loading mask: {mask_path}")
        
        try:
            mask_img = Image.open(mask_path).convert("L")
            mask_w, mask_h = mask_img.size
            
            # 检查 mask 尺寸是否与图像尺寸匹配
            if mask_h != h or mask_w != w:
                print(f"[DEBUG] Resizing mask from ({mask_w}, {mask_h}) to ({w}, {h})")
                mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)
            
            mask = np.array(mask_img, dtype=np.float32) / 255.0
            mask_bool = mask >= 0.5
            
            mask_coverage = np.sum(mask_bool) / mask_bool.size
            print(f"[INFO] Mask loaded: shape={mask_bool.shape}, coverage={mask_coverage*100:.1f}%")
        except Exception as e:
            print(f"[ERROR] Failed to load mask: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 6. 在 mask 区域内采样点
        mask_pixels = np.where(mask_bool)
        if len(mask_pixels[0]) == 0:
            print(f"[ERROR] Mask is empty (no white pixels)")
            return False
        
        # 随机采样或均匀采样
        num_mask_pixels = len(mask_pixels[0])
        samples_per_layer = num_samples
        if samples_per_layer > num_mask_pixels:
            samples_per_layer = num_mask_pixels
        
        indices = np.random.choice(num_mask_pixels, size=samples_per_layer, replace=False)
        v_samples = mask_pixels[0][indices]  # row (height)
        u_samples = mask_pixels[1][indices]  # col (width)
        
        print(f"[INFO] Sampling {samples_per_layer} points per layer from {num_mask_pixels} mask pixels")
        print(f"[INFO] Generating {num_depth_layers} depth layers from {near_depth} to {far_depth}")
        
        # 7. 将 2D 像素坐标投影到 3D 空间（多个深度层，形成立体视锥体）
        points_3d = []
        
        # 生成多个深度层
        depth_layers = np.linspace(near_depth, far_depth, num_depth_layers)
        
        for depth in depth_layers:
            for u, v in zip(u_samples, v_samples):
                # 归一化像素坐标（相机坐标系）
                x_norm = (u - cx) / fx
                y_norm = (v - cy) / fy
                
                # 在相机坐标系中的方向向量（nerfstudio 使用 -Z forward）
                dir_cam = np.array([x_norm, y_norm, -1.0])  # -Z forward
                dir_cam = dir_cam / np.linalg.norm(dir_cam)
                
                # 转换到世界坐标系
                dir_world = (c2w[:3, :3] @ dir_cam)
                
                # 从相机位置沿着方向投影到当前深度
                point_3d = camera_pos + dir_world * depth
                points_3d.append(point_3d)
        
        points_3d = np.array(points_3d)
        
        # 8. 添加相机位置点
        all_points = np.vstack([camera_pos.reshape(1, 3), points_3d])
        
        # 9. 保存为 PLY 文件
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建点云 mesh
        point_cloud = trimesh.PointCloud(vertices=all_points)
        point_cloud.export(str(output_ply_path))
        
        print(f"[INFO] Saved {len(all_points)} points to {output_ply_path}")
        print(f"[INFO]   - Camera position: 1 point")
        print(f"[INFO]   - Mask projection (3D volume): {len(points_3d)} points ({num_depth_layers} layers × {samples_per_layer} points)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to project mask to 3D: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Project camera position and mask to 3D space"
    )
    parser.add_argument(
        "--transforms_json",
        type=Path,
        required=True,
        help="Transforms.json file path"
    )
    parser.add_argument(
        "--view_index",
        type=int,
        required=True,
        help="View index (e.g., 50)"
    )
    parser.add_argument(
        "--output_ply_path",
        type=Path,
        required=True,
        help="Output PLY file path"
    )
    parser.add_argument(
        "--near_depth",
        type=float,
        default=1.0,
        help="Near plane depth from camera (default: 1.0)"
    )
    parser.add_argument(
        "--far_depth",
        type=float,
        default=10.0,
        help="Far plane depth from camera (default: 10.0)"
    )
    parser.add_argument(
        "--num_depth_layers",
        type=int,
        default=10,
        help="Number of depth layers (default: 10)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of points to sample from mask per layer (default: 10000)"
    )
    
    args = parser.parse_args()
    
    success = project_mask_to_3d(
        transforms_json_path=args.transforms_json,
        view_index=args.view_index,
        output_ply_path=args.output_ply_path,
        near_depth=args.near_depth,
        far_depth=args.far_depth,
        num_depth_layers=args.num_depth_layers,
        num_samples=args.num_samples,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

