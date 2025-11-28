#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List

from scipy.spatial.transform import Rotation

# 假设 camera_data 在您的路径中
from camera_data import CameraData, read_calibration_csv


def export_as_colmap(cameras: List[CameraData], output_folder: Path) -> None:
    camera_lines = ""
    image_lines = ""
    for camera_id, camera in enumerate(cameras):
        world_to_camera = Rotation.from_rotvec(-camera.rotation_axisangle)
        quat = world_to_camera.as_quat()
        tvec = -world_to_camera.as_matrix() @ camera.translation

        fx, fy, cx, cy = camera.fx_pixel, camera.fy_pixel, camera.cx_pixel, camera.cy_pixel
        camera_lines += f"{camera_id} PINHOLE {camera.width} {camera.height} {fx} {fy} {cx} {cy}\n"

        x, y, z, w = tuple(quat)
        tx, ty, tz = tuple(tvec)
        image_lines += f"{camera_id} {w} {x} {y} {z} {tx} {ty} {tz} {camera_id} {camera.name}\n\n"

    # Write intrinsics to cameras.txt
    with open(output_folder / "cameras.txt", "w") as f:
        f.write(camera_lines)

    # Write extrinsics to images.txt
    with open(output_folder / "images.txt", "w") as f:
        f.write(image_lines)

    # For completeness, write an empty points3D.txt file.
    with open(output_folder / "points3D.txt", "w") as f:
        f.write("# Empty file...\n")


# (!!! 新增函数 - 从 Canvas 移植并修改 !!!)
def export_as_nerf(cameras: List[CameraData], output_folder: Path) -> None:
    """
    将相机数据导出为 NeRF-studio 格式 (transforms.json)
    """
    
    # 用于将 COLMAP/CV 世界坐标系 转换为 NeRF/OpenGL 世界坐标系
    # P_nerf = M @ P_colmap
    # M = diag(1, -1, -1, 1)
    COLMAP_TO_NERF_WORLD = np.array([
        [1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, -1, 0], 
        [0, 0, 0, 1]
    ], dtype=np.float64)

    frames = []
    image_idx = 0  # 用于 camera_label

    for camera in cameras:
        # (移除了 'existing_images' 过滤)
            
        # 1. 获取 C2W (Camera-to-World) 变换 (在 COLMAP/CV 世界坐标系中)
        # 假设 camera.rotation_axisangle 和 camera.translation 是 C2W
        R_c2w_colmap = Rotation.from_rotvec(camera.rotation_axisangle).as_matrix()
        t_c2w_colmap = camera.translation
        
        # 组合成 4x4 矩阵
        T_c2w_colmap = np.eye(4, dtype=np.float64)
        T_c2w_colmap[:3, :3] = R_c2w_colmap
        T_c2w_colmap[:3, 3] = t_c2w_colmap

        # 2. 转换为 NeRF/OpenGL 世界坐标系下的 C2W 变换
        # T_c2w_nerf = M @ T_c2w_colmap
        T_c2w_nerf = COLMAP_TO_NERF_WORLD @ T_c2w_colmap
        
        # 3. 创建 NeRF 'frame' 字典
        frame = {
            # 您的示例要求 "OPENCV"。
            # 由于源数据是 "PINHOLE"，我们假设没有畸变。
            "camera_model": "OPENCV", 
            "camera_label": f"Cam{image_idx + 1:03d}", # (已修改) 格式: Cam001, Cam002...
            "w": camera.width,
            "h": camera.height,
            "fl_x": camera.fx_pixel,
            "fl_y": camera.fy_pixel,
            "cx": camera.cx_pixel,
            "cy": camera.cy_pixel,
            "k1": 0.0,  # 假设没有畸变
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "transform_matrix": T_c2w_nerf.tolist(),
            # (更新 file_path 以匹配 NeRF 约定)
            "file_path": f"images/{camera.name}" 
        }
        frames.append(frame)
        image_idx += 1

    # 4. 创建最终的 JSON 结构
    nerf_data = {
        "ply_file_path": "sparse_pcd.ply", # (已添加)
        "frames": frames
    }

    # 5. 写入文件
    output_file = output_folder / "transforms.json"
    with open(output_file, "w") as f:
        json.dump(nerf_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path('/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/calibration.csv'))
    parser.add_argument("--output_dir",type=Path, default=Path('/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x'))
    
    # (!!! 新增参数 !!!)
    parser.add_argument(
        "--export_nerf",
        default=True,
        type=str,
        help="Also export a transforms.json for NeRF-based methods"
    )
    args = parser.parse_args()

    cameras = read_calibration_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_as_colmap(cameras, args.output_dir)
    print(f"Exported COLMAP files to {args.output_dir}")

    # (!!! 新增逻辑 !!!)
    # 如果指定了 --export_nerf, 则同时导出 transforms.json
    if args.export_nerf:
        export_as_nerf(cameras, args.output_dir)
        print(f"Exported NeRF transforms.json to {args.output_dir}")


if __name__ == "__main__":
    main()


