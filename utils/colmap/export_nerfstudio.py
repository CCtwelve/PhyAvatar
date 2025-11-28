#!/usr/bin/env python3
'''
Cam127/460.png 

'''
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Set
import shutil
import csv
import os
from scipy.spatial.transform import Rotation
from padded import padding
# ... (CameraData 类定义保持不变) ...
class CameraData:
    def __init__(self, name, width, height, rotation_axisangle, translation, focal_length, principal_point):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.rotation_axisangle = np.array(rotation_axisangle, dtype=np.float64)
        self.translation = np.array(translation, dtype=np.float64)
        self.focal_length = np.array(focal_length, dtype=np.float64)
        self.principal_point = np.array(principal_point, dtype=np.float64)
    # ... (属性定义保持不变) ...
    @property
    def fx_pixel(self) -> float:
        return self.focal_length[0]
    @property
    def fy_pixel(self) -> float:
        return self.focal_length[1]
    @property
    def cx_pixel(self) -> float:
        return self.principal_point[0]
    @property
    def cy_pixel(self) -> float:
        return self.principal_point[1]


# ... (read_calibration_csv 函数保持不变) ...
def read_calibration_csv(input_csv_path: Path) -> List[CameraData]:
    cameras = []
    try:
        with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                camera = CameraData(
                    name=row["name"],
                    width=int(row["w"]),
                    height=int(row["h"]),
                    rotation_axisangle=np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])]),
                    translation=np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])]),
                    focal_length=np.array([float(row["fx"]), float(row["fy"])]),
                    principal_point=np.array([float(row["px"]), float(row["py"])]),
                )
                cameras.append(camera)
    except FileNotFoundError:
        print(f"错误：找不到校准文件：{input_csv_path}")
        return []
    except KeyError as e:
        print(f"错误：CSV 文件中缺少必需的列：{e}")
        return []
    return cameras

# ❗️ 修正 1: 函数签名现在需要 view_list_names 和 pose_filename
def export_as_nerf(input_csv_path: Path, output_folder: Path, pose_filename: str, view_list_names: Set[str]) -> None:
    import cv2 as cv
    """
    Args:
        ... (其他参数)
        pose_filename (str): 姿态文件名 (例如 "460")。
        view_list_names (Set[str]): 要包含的相机名称集合。
    """
    # ... (COLMAP_TO_NERF_WORLD 保持不变) ...
    COLMAP_TO_NERF_WORLD = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

    frames = []
    with open(input_csv_path, "r", newline = "", encoding = 'utf-8') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            cam_names=row['name']

            # ❗️ 修正 2: 检查 cam_names 是否在我们要的列表中
            if cam_names not in view_list_names:
                continue # 如果不在，跳过此相机

            # ... (extr_mat 和 T_c2w_nerf 计算保持不变) ...
            img_widths = int(row['w'])
            img_heights = int(row['h']) 
            extr_mat = np.identity(4, np.float32)
            extr_mat[:3, :3] = cv.Rodrigues(np.array([float(row['rx']), float(row['ry']), float(row['rz'])], np.float32))[0]
            extr_mat[:3, 3] = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
            extr_mat[:3,1:3] *= -1
            T_c2w_nerf = extr_mat
            
            # ... (padding 计算保持不变) ...
            original_w = img_widths   
            original_h = img_heights 
            target_w = 1024
            target_h = 1024
            original_fl_x = float(row['fx']) * float(row['w'])
            original_fl_y = float(row['fy']) * float(row['h'])
            original_cx = float(row['px']) * float(row['w'])
            original_cy = float(row['py']) * float(row['h'])
            offset_x = (target_w - original_w) // 2
            offset_y = (target_h - original_h) // 2 

            frame = {
                "camera_model": "OPENCV", 
                "camera_label": cam_names,   
                "w": target_w,
                "h": target_h,
                "fl_x": original_fl_x,
                "fl_y": original_fl_y,
                "cx": original_cx + offset_x,
                "cy": original_cy + offset_y,
                "k1": 0.0, 
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "transform_matrix": T_c2w_nerf.tolist(),
                
                # ❗️ 修正 3: 使用 pose_filename (例如 "460")
                "file_path": f"images/{cam_names}/{pose_filename}.jpg",
                "mask_path": f"fmasks/{cam_names}/{pose_filename}.png"
            }
            frames.append(frame)

    # ... (nerf_data 和文件写入保持不变) ...
    nerf_data = {"frames": frames}
    output_folder.mkdir(parents=True, exist_ok=True) 
    output_file = output_folder / "transforms.json"
    with open(output_file, "w") as f:
        json.dump(nerf_data, f, indent=4)


def main():
    # ... (argparse 定义保持不变) ...
    parser = argparse.ArgumentParser(description="复制特定姿态的图像/掩码，并为 NeRF 导出筛选后的相机校准数据。")
    parser.add_argument("--view_list", type=int, nargs='+', default=[127,40,95,], help="要处理的相机视图 ID 列表。")
    parser.add_argument("--pose", type=str, default="460", help="要提取的姿态编号 (例如 '460')。")
    parser.add_argument("--csv", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/calibration.csv"), help="输入的相机校准 CSV 文件路径。")
    parser.add_argument("--input_images_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/rgbs"), help="源 RGB 图像的根目录。")
    parser.add_argument("--input_masks_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/masks"), help="源掩码的根目录。")
    parser.add_argument("--out_images_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_3_2_17/images"), help="目标 RGB 图像的根目录。")
    parser.add_argument("--out_fmasks_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_3_2_17/fmasks"), help="目标掩码的根目录。")
    parser.add_argument("--output_dir", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_3_2_17"), help="transforms.json 文件的输出目录。")
    parser.add_argument("--export_nerf", default=True, help="导出 transforms.json 供 NeRF-based 方法使用 (默认为 True)。")
    args = parser.parse_args()

    # --- 1. 读取和筛选相机 ---
    all_cameras = read_calibration_csv(args.csv)
    if not all_cameras:
        print(f"未能从 {args.csv} 读取相机数据，正在退出。")
        return

    # 这就是我们需要的过滤器
    view_list_names = {f"Cam{view:03d}" for view in args.view_list}
    filtered_cameras = [cam for cam in all_cameras if cam.name in view_list_names]
    
    # ... (print 语句保持不变) ...
    print(f"从 {args.csv} 读取了 {len(all_cameras)} 个相机。")
    print(f"根据 view_list 筛选后，保留 {len(filtered_cameras)} 个相机。")
    if len(filtered_cameras) == 0:
        print(f"警告：筛选结果为 0。请检查 calibration.csv 中的名称是否与 'Cam006', 'Cam021' ... 匹配")

    # --- 2. 复制图像和掩码 ---
    pose_str_padded = args.pose.zfill(6) # "460" -> "000460"
    print(f"\n开始复制文件 (姿态: {args.pose}, 源填充: {pose_str_padded})...")
    
    # ... (文件复制循环保持不变) ...
    args.out_images_path.mkdir(parents=True, exist_ok=True)
    args.out_fmasks_path.mkdir(parents=True, exist_ok=True)
    for view in args.view_list:
        cam_name = f"Cam{view:03d}" # 6 -> "Cam006"
        src_img = args.input_images_path / cam_name / f"{cam_name}_rgb{pose_str_padded}.jpg"
        dest_img = args.out_images_path /cam_name/ f"{args.pose}.jpg"
        os.makedirs(args.out_images_path /cam_name, exist_ok=True)
        if src_img.exists():
            shutil.copy(src_img, dest_img)
        else:
            print(f"  [!] 警告: 找不到源图像: {src_img}")
        src_mask = args.input_masks_path / cam_name / f"{cam_name}_mask{pose_str_padded}.png"
        dest_mask = args.out_fmasks_path /cam_name/ f"{args.pose}.png"
        os.makedirs(args.out_fmasks_path /cam_name, exist_ok=True)
        if src_mask.exists():
            shutil.copy(src_mask, dest_mask)
        else:
            print(f"  [!] 警告: 找不到源掩码: {src_mask}")
    print("文件复制完成。")

    # --- 3. 导出 NeRF 数据 ---
    if args.export_nerf:
        if not filtered_cameras:
            print("没有筛选后的相机可用于导出 NeRF.json。")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            
            # ❗️ 修正 4: 将 view_list_names 和 args.pose 传递给函数
            export_as_nerf(args.csv, args.output_dir, args.pose, view_list_names)
            
            print(f"\n已将 {len(filtered_cameras)} 个相机的 NeRF transforms.json 导出到 {args.output_dir}")

    
        # --- 4. 填充复制的图像和掩码 ---
    print("\n开始填充已复制的图像...")
    padding(args.out_images_path, target_size=(1024, 1024), fill_color=(0, 0, 0))
    
    print("\n开始填充已复制的掩码...")
    padding(args.out_fmasks_path, target_size=(1024, 1024), fill_color=(0, 0, 0)) 

if __name__ == "__main__":
    main()