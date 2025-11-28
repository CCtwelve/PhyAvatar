''' csv to transform.json
生成的文件路径格式与export_nerfstudio不同


'''
#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Set
import shutil
import csv
import os
from scipy.spatial.transform import Rotation
from padded import padding,batch_convert_to_webp 

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

# ❗️ 函数签名不变 (它需要 reindex_mapping)
def export_as_nerf(input_csv_path: Path, output_folder: Path, pose_filename: str, view_list_names: Set[str], reindex_mapping: dict) -> None:
    import cv2 as cv
    """
    Args:
        ... (其他参数)
        reindex_mapping (dict): 字典，将原始 cam_name 映射到新的 re-indexed cam_name.
    """
    COLMAP_TO_NERF_WORLD = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

    frames = []
    with open(input_csv_path, "r", newline = "", encoding = 'utf-8') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # ❗️ 这是原始标签
            original_cam_label = row['name'] # e.g., "Cam127"

            # 检查 cam_names 是否在我们要的列表中
            if original_cam_label not in view_list_names:
                continue # 如果不在，跳过此相机

            # ❗️ 获取新的 re-indexed 标签 (现在是 "00", "01" 等)
            new_cam_label = reindex_mapping[original_cam_label] # e.g., "10"

            # 构建 COLMAP 坐标系下的 C2W 变换矩阵
            img_widths = int(row['w'])
            img_heights = int(row['h']) 
            T_c2w_colmap = np.identity(4, dtype=np.float64)
            T_c2w_colmap[:3, :3] = cv.Rodrigues(np.array([float(row['rx']), float(row['ry']), float(row['rz'])], dtype=np.float64))[0]
            T_c2w_colmap[:3, 3] = np.array([float(row['tx']), float(row['ty']), float(row['tz'])], dtype=np.float64)
            
            # 转换为 NeRF/OpenGL 坐标系：T_c2w_nerf = COLMAP_TO_NERF_WORLD @ T_c2w_colmap
            # 这会同时转换旋转和平移部分
            T_c2w_nerf = COLMAP_TO_NERF_WORLD @ T_c2w_colmap
            
            # ... (padding 计算保持不变) ...
            original_w = img_widths   
            original_h = img_heights 
            target_w = 1024
            target_h = 1024
            original_fl_x = float(row['fx']) * float(row['w'])
            original_fl_y = float(row['fy']) * float(row['h'])
            original_cx = float(row['px']) * float(row['w'])
            original_cy = float(row['py']) * float(row['h'])
            # 使用浮点数计算偏移量以保持精度（图像居中填充）
            offset_x = (target_w - original_w) / 2.0
            offset_y = (target_h - original_h) / 2.0 

            # ❗️ 修正 4: 更新 frame 字典
            frame = {
                "camera_model": "OPENCV", 
                "camera_label": new_cam_label,         # ❗️ 使用新的 "00", "01", ...
                "orint_camera_label": original_cam_label, # ❗️ 添加原始 "Cam127"
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
                
                # ❗️ 修正 5: 使用 new_cam_label (例如 "00") 和 pose_filename
                "file_path": f"images/{new_cam_label}/{pose_filename}.jpg",
                "mask_path": f"fmasks/{new_cam_label}/{pose_filename}.png"
            }
            frames.append(frame)

    # ... (nerf_data 和文件写入保持不变) ...
    nerf_data = {"ply_file_path": "sparse_pcd.ply","frames": frames}
    output_folder.mkdir(parents=True, exist_ok=True) 
    output_file = output_folder / "transforms.json"
    with open(output_file, "w") as f:
        json.dump(nerf_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="复制特定姿态的图像/掩码，并为 NeRF 导出筛选后的相机校准数据。")
    parser.add_argument("--view_list", type=int, nargs='+', default=[127,40,95,126,128,6, 8, 21, 23, 110, 112, 135, 136, 143, 144, 151, 152, 159, 160], help="要处理的相机视图 ID 列表。")
    parser.add_argument("--json_list", type=int, nargs='+', default=[127,40,95,126,128,6, 8, 21, 23, 110, 112, 135, 136, 143, 144, 151, 152, 159, 160], help="要处理的相机视图 ID 列表。")
    parser.add_argument("--pose", type=str, default="460", help="要提取的姿态编号 (例如 '460')。")
    parser.add_argument("--csv", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/calibration.csv"), help="输入的相机校准 CSV 文件路径。")
    parser.add_argument("--input_images_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/rgbs"), help="源 RGB 图像的根目录。")
    parser.add_argument("--input_masks_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/masks"), help="源掩码的根目录。")
    parser.add_argument("--out_images_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/images"), help="目标 RGB 图像的根目录。")
    parser.add_argument("--out_fmasks_path", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/fmasks"), help="目标掩码的根目录。")
    parser.add_argument("--output_dir", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17"), help="transforms—_change.json 文件的输出目录。")
    parser.add_argument("--export_nerf", default=True, help="导出 transforms.json 供 NeRF-based 方法使用 (默认为 True)。")
    args = parser.parse_args()


    # front_view = [5,6,7,8,21,22,23,24,109,110,111,112,126,127,128,135,136,143,144]
    # side_view = [37,38,39,40,93,94,95,96]
    # back_view = [53,54,55,56,77,78,79,80,151,152,159,160] 

    # half_view = [131,139,147,155, 132,140,148,156]
    # head_view = [121,122,123,124,125,133,141,149,157,134,142,150,158]

        # 合并 view_list
    # args.view_list = front_view + side_view + back_view 
    # args.view_list = [6, 8, 21, 23, 110, 112, 126, 127, 128, 135, 136, 143, 144, 151, 152, 159, 160]
    
    # --- 1. 读取和筛选相机 ---
    all_cameras = read_calibration_csv(args.csv)
    if not all_cameras:
        print(f"未能从 {args.csv} 读取相机数据，正在退出。")
        return
    

    
    # ❗️ 修正 7: 创建 re-index 映射
    # 7a. 获取所有原始相机名称的 *有序* 列表
    view_cam_names = [f"Cam{view:03d}" for view in args.view_list]
    json_cam_names = [f"Cam{view:03d}" for view in args.json_list]
    combined_cam_names = []
    for cam_name in view_cam_names + json_cam_names:
        if cam_name not in combined_cam_names:
            combined_cam_names.append(cam_name)
    reindex_mapping = {cam_name: f"{i:02d}" for i, cam_name in enumerate(combined_cam_names)}
    
    view_list_names = set(view_cam_names)
    json_list_names = set(json_cam_names)
    
    filtered_cameras = [cam for cam in all_cameras if cam.name in json_list_names]
    
    print(f"从 {args.csv} 读取了 {len(all_cameras)} 个相机。")
    print(f"根据 json_list 筛选后，保留 {len(filtered_cameras)} 个相机。")
    if len(filtered_cameras) == 0:
        print(f"警告：筛选结果为 0。请检查 .csv 中的名称是否与 'Cam005', 'Cam006' ... 匹配")

    # --- 2. 复制图像和掩码 ---
    pose_str_padded = args.pose.zfill(6) # "460" -> "000460" (用于读取)
    
    # 定义新的目标文件名 (用于存储)
    dest_pose_filename = "000000"
    
    print(f"\n开始复制文件 (姿态: {args.pose}, 目标: {dest_pose_filename})...")
    
    args.out_images_path.mkdir(parents=True, exist_ok=True)
    args.out_fmasks_path.mkdir(parents=True, exist_ok=True)
    
    # 循环遍历原始 view_list 以查找文件
    for view in args.view_list:
        original_cam_label = f"Cam{view:03d}" # e.g., "Cam127"
        
        if original_cam_label not in reindex_mapping:
            print(f" [!] 警告: {original_cam_label} 在 view_list 中但不在 reindex_mapping 中。跳过。")
            continue
            
        # 获取新的 re-indexed 标签 (现在是 "00", "01" 等)
        new_cam_label = reindex_mapping[original_cam_label] # e.g., "10"

        # --- 复制图像 ---
        src_img = args.input_images_path / original_cam_label / f"{original_cam_label}_rgb{pose_str_padded}.jpg"
        
        # 目标路径使用 new_cam_label (例如 "00") 和 dest_pose_filename
        dest_img_dir = args.out_images_path / new_cam_label
        dest_img = dest_img_dir / f"{dest_pose_filename}.jpg"
        
        os.makedirs(dest_img_dir, exist_ok=True)
        if src_img.exists():
            shutil.copy(src_img, dest_img)
        else:
            print(f" 	[!] 警告: 找不到源图像: {src_img}")

        # --- 复制掩码 ---
        src_mask = args.input_masks_path / original_cam_label / f"{original_cam_label}_mask{pose_str_padded}.png"
        
        # 目标路径使用 new_cam_label (例如 "00") 和 dest_pose_filename
        dest_mask_dir = args.out_fmasks_path / new_cam_label
        dest_mask = dest_mask_dir / f"{dest_pose_filename}.png"
        
        os.makedirs(dest_mask_dir, exist_ok=True)
        if src_mask.exists():
            shutil.copy(src_mask, dest_mask)
        else:
            print(f" 	[!] 警告: 找不到源掩码: {src_mask}")
    
    print("文件复制完成。")

    # --- 3. 导出 NeRF 数据 ---
    if args.export_nerf:
        if not filtered_cameras:
            print("没有筛选后的相机可用于导出 NeRF.json。")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 将 reindex_mapping 和 dest_pose_filename 传递给函数
            export_as_nerf(args.csv, args.output_dir, dest_pose_filename, json_list_names, reindex_mapping)
            
            print(f"\n已将 {len(filtered_cameras)} 个相机的 NeRF transforms.json 导出到 {args.output_dir}")
    
    # --- 4. 填充复制的图像和掩码 ---
    print("\n开始填充已复制的图像...")
    padding(args.out_images_path, target_size=(1024, 1024), fill_color=(0, 0, 0))
    
    print("\n开始填充已复制的掩码...")
    padding(args.out_fmasks_path, target_size=(1024, 1024), fill_color=(0, 0, 0)) 

    print("\n所有处理已完成。")

if __name__ == "__main__":
    main()