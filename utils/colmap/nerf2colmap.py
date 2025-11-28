import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


NERF_TO_COLMAP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

path = Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_7_2_48")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Nerfstudio 转 COLMAP：将 transforms.json 转回 COLMAP 稀疏模型"
    )
    parser.add_argument(
        "--nerfstudio-root",
        type=Path,
        default=path,
        help="包含 transforms.json 与 images/ 的 Nerfstudio 目录",
    )
    parser.add_argument(
        "--transforms-name",
        type=str,
        default="transforms.json",
        help="Nerfstudio 导出的 json 文件名",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=path / "colmap_2_nerf_7_2_48",
        help="COLMAP 项目的输出根目录（会创建 images / sparse/0 等目录）",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="默认创建符号链接，指定后改为复制图片",
    )
    parser.add_argument(
        "--export-binary",
        action="store_true",
        help="写出 text 模型后，额外调用 colmap model_converter 生成 binary 模型",
    )
    parser.add_argument(
        "--colmap-bin",
        type=str,
        default="colmap",
        help="colmap 可执行文件（--export-binary 时使用）",
    )
    parser.add_argument(
        "--reconstruct",
        default=True,
        help="运行 COLMAP 特征提取、匹配和三角化以生成稀疏点云",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="使用 GPU 进行特征提取和匹配（默认启用）",
    )
    return parser.parse_args()


def quaternion_from_matrix(R: np.ndarray) -> Tuple[float, float, float, float]:
    """COLMAP 期望的 (qw, qx, qy, qz)"""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if idx == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return qw, qx, qy, qz


def invert_pose(mat: np.ndarray) -> np.ndarray:
    """返回 4x4 w2c"""
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def ensure_images(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=True)


def group_cameras(frames: List[dict]) -> Dict[Tuple, int]:
    mapping: Dict[Tuple, int] = {}
    for frame in frames:
        params = (
            int(frame["w"]),
            int(frame["h"]),
            float(frame["fl_x"]),
            float(frame["fl_y"]),
            float(frame["cx"]),
            float(frame["cy"]),
        )
        if params not in mapping:
            mapping[params] = len(mapping) + 1
    return mapping


def write_cameras_txt(path: Path, mapping: Dict[Tuple, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(mapping)))
        for params, cam_id in mapping.items():
            width, height, fl_x, fl_y, cx, cy = params
            line = f"{cam_id} PINHOLE {width} {height} {fl_x} {fl_y} {cx} {cy}\n"
            f.write(line)


def normalize_image_name(file_path: str) -> str:
    """
    将不同来源的路径规范化为 COLMAP 数据库使用的形式：
    - 统一使用 "/" 作为分隔符，去掉开头的 "./"
    - 移除诸如 "images/"、"images_alpha/" 等前缀
    - Nerfstudio 的 images_alpha 目录是 0-based，下游真实目录通常是 1-based，这里自动 +1
    - 将 .png 扩展名统一映射为 .jpg（COLMAP 输出通常是 jpg）
    """
    if not file_path:
        return file_path

    # 统一分隔符并去掉开头的 "./"
    normalized = file_path.replace("\\", "/").lstrip("./")

    # 记录是否来自 0-based 的 images_alpha
    zero_index_prefixes = ("images_alpha/",)
    strip_prefixes = ("images/", "images_alpha/", "rgb/", "rgba/", "images_rgb/")
    came_from_zero_index = False
    for prefix in strip_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            if prefix in zero_index_prefixes:
                came_from_zero_index = True
            break

    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return normalized

    # 如果目录部分是数字且来自 0-based，则 +1 并保持宽度
    if came_from_zero_index and len(parts) >= 2 and parts[0].isdigit():
        width = len(parts[0])
        idx = int(parts[0])
        parts[0] = f"{idx + 1:0{width}d}"

    # 将 .png 转成 .jpg 以匹配数据库中的文件
    filename = parts[-1]
    stem, ext = os.path.splitext(filename)
    if ext.lower() == ".png":
        parts[-1] = f"{stem}.jpg"

    return "/".join(parts)


def write_images_txt(
    path: Path,
    frames: List[dict],
    camera_mapping: Dict[Tuple, int],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, frame in enumerate(frames, start=1):
            params = (
                int(frame["w"]),
                int(frame["h"]),
                float(frame["fl_x"]),
                float(frame["fl_y"]),
                float(frame["cx"]),
                float(frame["cy"]),
            )
            camera_id = camera_mapping[params]
            c2w_nerf = np.array(frame["transform_matrix"], dtype=np.float64)
            # NeRF(OpenGL) -> COLMAP(OpenCV) 通过右乘轴变换矩阵
            c2w_colmap = c2w_nerf @ NERF_TO_COLMAP
            w2c_colmap = invert_pose(c2w_colmap)
            rot = w2c_colmap[:3, :3]
            trans = w2c_colmap[:3, 3]
            qw, qx, qy, qz = quaternion_from_matrix(rot)
            # 规范化图像名称，确保与文件系统一致
            image_name = normalize_image_name(frame["file_path"])
            line = (
                f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{trans[0]:.9f} {trans[1]:.9f} {trans[2]:.9f} "
                f"{camera_id} {image_name}\n"
            )
            f.write(line)
            f.write("\n")


def write_points3d_txt(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")


def convert_model(colmap_bin: str, input_path: Path, output_path: Path, output_type: str) -> None:
    """
    使用 COLMAP 的 model_converter 在不同格式之间转换模型。
    output_type: TXT / BIN / PLY / ...
    """
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin,
        "model_converter",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        output_type,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_binary_model(colmap_bin: str, text_dir: Path, bin_dir: Path) -> None:
    convert_model(colmap_bin, text_dir, bin_dir, "BIN")


def run_feature_extraction(
    colmap_bin: str,
    database_path: Path,
    image_path: Path,
    use_gpu: bool = True,
) -> None:
    """运行 COLMAP 特征提取"""
    use_gpu_int = 1 if use_gpu else 0
    cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--SiftExtraction.peak_threshold",
        "0.0005",
        "--ImageReader.single_camera",
        "0",
        "--SiftExtraction.edge_threshold",
        "300",
        "--SiftExtraction.use_gpu",
        str(use_gpu_int),
        "--SiftExtraction.first_octave",
        "-1",
        "--SiftExtraction.estimate_affine_shape",
        "true",
        "--SiftExtraction.domain_size_pooling",
        "true",
    ]
    print("Running feature extraction:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_feature_matching(
    colmap_bin: str,
    database_path: Path,
    use_gpu: bool = True,
) -> None:
    """运行 COLMAP 特征匹配"""
    use_gpu_int = 1 if use_gpu else 0
    cmd = [
        colmap_bin,
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--SiftMatching.use_gpu",
        str(use_gpu_int),
        "--SiftMatching.min_num_inliers",
        "5",
    ]
    print("Running feature matching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_point_triangulator(
    colmap_bin: str,
    database_path: Path,
    image_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """运行 COLMAP 点三角化"""
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin,
        "point_triangulator",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
    ]
    print("Running point triangulator:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_ply(colmap_bin: str, input_path: Path, output_path: Path) -> None:
    """将 COLMAP 模型导出为 PLY 格式"""
    cmd = [
        colmap_bin,
        "model_converter",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        "PLY",
    ]
    print("Running model converter to PLY:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def rename_subdirectories_and_update_paths(base_path: Path, frames: List[dict]) -> None:
    """
    重命名 images、images_alpha 和 fmasks 目录下的子目录（将数字减1），
    并更新 transforms.json 中的 file_path 和 mask_path。
    
    关键：从小到大排序重命名，避免命名冲突。
    例如：先重命名 01->00，然后 02->01，最后 03->02。
    """
    dirs_to_process = ["images", "images_alpha", "fmasks"]
    rename_operations = []  # 存储 (dir_name, old_name, new_name, old_path, new_path)
    
    # 第一步：收集所有需要重命名的目录
    for dir_name in dirs_to_process:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            continue
        
        # 获取所有子目录
        for subdir in dir_path.iterdir():
            if not subdir.is_dir():
                continue
            
            subdir_name = subdir.name
            # 检查是否是数字格式（如 "01", "02", "001" 等）
            if subdir_name.isdigit():
                try:
                    old_value = int(subdir_name)
                    if old_value > 0:  # 只处理大于0的数字
                        new_value = old_value - 1
                        # 保持原有的零填充格式
                        width = len(subdir_name)
                        new_name = f"{new_value:0{width}d}"
                        rename_operations.append((
                            dir_name,
                            subdir_name,
                            new_name,
                            subdir,
                            dir_path / new_name
                        ))
                except ValueError:
                    continue
    
    if not rename_operations:
        print("未找到需要重命名的子目录")
        return
    
    # 第二步：按数字从小到大排序（先 01->00，再 02->01，以此类推）
    # 排序键：(dir_name, old_value) 确保同一目录内从小到大，不同目录按名称排序
    rename_operations.sort(key=lambda x: (x[0], int(x[1])))
    
    # 第三步：执行重命名
    rename_mapping = {}  # 存储旧路径到新路径的映射（用于更新 transforms.json）
    for dir_name, old_name, new_name, old_path, new_path in rename_operations:
        if old_path.exists():
            if new_path.exists():
                print(f"警告: 目标目录已存在，跳过: {dir_name}/{old_name} -> {dir_name}/{new_name}")
            else:
                old_path.rename(new_path)
                print(f"已重命名: {dir_name}/{old_name} -> {dir_name}/{new_name}")
                # 存储映射关系（用于更新路径）
                key = f"{dir_name}/{old_name}"
                rename_mapping[key] = f"{dir_name}/{new_name}"
        else:
            print(f"警告: 源目录不存在: {dir_name}/{old_name}")
    
    if not rename_mapping:
        print("没有成功重命名任何目录")
        return
    
    # 第四步：更新 transforms.json 中的路径
    updated_count = 0
    for frame in frames:
        # 更新 file_path
        if "file_path" in frame:
            file_path = frame["file_path"]
            # 统一使用正斜杠处理
            file_path_normalized = file_path.replace("\\", "/")
            for old_key, new_key in rename_mapping.items():
                dir_name = old_key.split('/')[0]  # 如 "images", "images_alpha", "fmasks"
                old_dir = old_key.split('/')[-1]  # 如 "01", "02"
                new_dir = new_key.split('/')[-1]  # 如 "00", "01"
                
                # 匹配路径格式：dir_name/old_dir/...（确保匹配完整的目录前缀）
                pattern_str = f"{dir_name}/{old_dir}/"
                if pattern_str in file_path_normalized:
                    # 替换所有出现的地方
                    file_path_normalized = file_path_normalized.replace(pattern_str, f"{dir_name}/{new_dir}/")
                    # 保持原有的路径分隔符格式
                    if "\\" in file_path:
                        file_path_normalized = file_path_normalized.replace("/", "\\")
                    frame["file_path"] = file_path_normalized
                    updated_count += 1
                    break
        
        # 更新 mask_path（如果存在）
        if "mask_path" in frame:
            mask_path = frame["mask_path"]
            # 统一使用正斜杠处理
            mask_path_normalized = mask_path.replace("\\", "/")
            for old_key, new_key in rename_mapping.items():
                dir_name = old_key.split('/')[0]  # 如 "fmasks"
                old_dir = old_key.split('/')[-1]  # 如 "01", "02"
                new_dir = new_key.split('/')[-1]  # 如 "00", "01"
                
                # 匹配路径格式：dir_name/old_dir/...（确保匹配完整的目录前缀）
                pattern_str = f"{dir_name}/{old_dir}/"
                if pattern_str in mask_path_normalized:
                    # 替换所有出现的地方
                    mask_path_normalized = mask_path_normalized.replace(pattern_str, f"{dir_name}/{new_dir}/")
                    # 保持原有的路径分隔符格式
                    if "\\" in mask_path:
                        mask_path_normalized = mask_path_normalized.replace("/", "\\")
                    frame["mask_path"] = mask_path_normalized
                    updated_count += 1
                    break
    
    print(f"已更新 {updated_count} 个路径引用")


def update_images_txt_from_database(
    database_path: Path, images_txt_path: Path
) -> None:
    """
    根据数据库中的图像信息，完全重新生成 images.txt 文件。
    确保图像 ID 和名称都与数据库一致，这对于 point_triangulator 至关重要。
    
    数据库中的图像 ID 和名称是 feature_extractor 根据实际文件系统生成的，
    必须与 images.txt 中的完全匹配，否则 point_triangulator 无法关联特征点。
    """
    # 从数据库读取图像信息（按 image_id 排序）
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
    )
    db_images = [
        {"image_id": row[0], "name": row[1], "camera_id": row[2]}
        for row in cursor.fetchall()
    ]
    if not db_images:
        conn.close()
        raise ValueError("数据库中未找到图像记录")

    # 读取每张图像的特征点坐标（COLMAP 存储为 rows x cols，前两个维度是像素坐标）
    cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
    db_keypoints = {}
    for image_id, n_rows, n_cols, blob in cursor.fetchall():
        if not blob or n_rows == 0 or n_cols < 2:
            continue
        points = np.frombuffer(blob, dtype=np.float32)
        expected = n_rows * n_cols
        if points.size != expected:
            print(
                f"警告: 图像 {image_id} 的特征点数据大小不匹配 "
                f"(rows={n_rows}, cols={n_cols}, 实际={points.size})，跳过"
            )
            continue
        points = points.reshape((n_rows, n_cols))[:, :2]
        db_keypoints[image_id] = points
    conn.close()
    
    # 读取现有的 images.txt，建立图像名称到位姿的映射
    # 使用规范化后的图像名称作为键，以处理路径差异
    name_to_pose = {}
    with open(images_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过注释行和空行
        if not line or line.startswith("#"):
            i += 1
            continue
        
        # 解析图像行
        parts = line.split()
        if len(parts) >= 10:
            # 提取图像名称（最后一个字段）并规范化
            image_name_raw = parts[9]
            image_name = normalize_image_name(image_name_raw)
            # 存储位姿信息（除了 image_id 和 name）
            pose_data = {
                "qw": parts[1],
                "qx": parts[2],
                "qy": parts[3],
                "qz": parts[4],
                "tx": parts[5],
                "ty": parts[6],
                "tz": parts[7],
                "camera_id": parts[8],
            }
            # 同时存储原始名称和规范化名称的映射
            name_to_pose[image_name] = pose_data
            if image_name != image_name_raw:
                name_to_pose[image_name_raw] = pose_data
        
        i += 1
        # 跳过下一行（POINTS2D 行）
        if i < len(lines):
            i += 1
    
    # 根据数据库中的图像信息，重新生成 images.txt
    # 使用数据库中的图像 ID 和名称，但保留原有的位姿信息
    with open(images_txt_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        matched_count = 0
        for db_img in db_images:
            db_name = db_img["name"]
            db_image_id = db_img["image_id"]
            db_camera_id = db_img["camera_id"]
            
            # 规范化数据库中的图像名称
            db_name_normalized = normalize_image_name(db_name)
            
            # 尝试匹配图像名称
            # 首先尝试规范化后的名称
            pose = name_to_pose.get(db_name_normalized)
            
            # 如果失败，尝试原始名称
            if pose is None:
                pose = name_to_pose.get(db_name)
            
            # 如果还是失败，尝试只匹配文件名（去掉路径）
            if pose is None:
                db_name_basename = Path(db_name_normalized).name
                for name, p in name_to_pose.items():
                    if Path(normalize_image_name(name)).name == db_name_basename:
                        pose = p
                        break
            
            if pose:
                # 使用数据库中的 camera_id（如果数据库中有的话），否则使用原有的
                camera_id = db_camera_id if db_camera_id else pose["camera_id"]
                line = (
                    f"{db_image_id} {pose['qw']} {pose['qx']} {pose['qy']} {pose['qz']} "
                    f"{pose['tx']} {pose['ty']} {pose['tz']} {camera_id} {db_name}\n"
                )
                f.write(line)
                keypoints = db_keypoints.get(db_image_id)
                if keypoints is None or len(keypoints) == 0:
                    f.write("\n")
                else:
                    triplets = " ".join(
                        f"{x:.6f} {y:.6f} -1" for x, y in keypoints
                    )
                    f.write(f"{triplets}\n")
                matched_count += 1
            else:
                print(f"警告: 数据库中图像 '{db_name}' (ID: {db_image_id}) 在原有 images.txt 中未找到位姿信息，跳过")
    
    print(
        f"已重新生成 images.txt，使用数据库中的图像 ID 和名称（匹配了 {matched_count}/{len(db_images)} 个图像，"
        f"填充了 {len(db_keypoints)} 张图像的 POINTS2D）"
    )


def main() -> None:
    args = parse_args()
    transforms_path = args.nerfstudio_root / args.transforms_name
    images_src = args.nerfstudio_root / "images"

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms json 不存在: {transforms_path}")
    if not images_src.exists():
        raise FileNotFoundError(f"未找到 Nerfstudio images 目录: {images_src}")

    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("transforms.json 中缺少 frames 列表")
    
    # 将每个 frame 的 camera_label 减1（01变为00，02变为01等）
    for frame in frames:
        if "camera_label" in frame:
            try:
                # 将字符串转换为整数，减1，再转换回字符串，保持原有格式
                current_label = frame["camera_label"]
                if isinstance(current_label, str) and current_label.isdigit():
                    new_value = int(current_label) - 1
                    # 保持原有的零填充格式
                    width = len(current_label)
                    frame["camera_label"] = f"{new_value:0{width}d}"
                elif isinstance(current_label, (int, float)):
                    frame["camera_label"] = int(current_label) - 1
            except (ValueError, TypeError) as e:
                print(f"警告: 无法处理 camera_label '{frame.get('camera_label')}': {e}")
    
    # 重命名 images、images_alpha 和 fmasks 目录下的子目录，并更新路径
    print("\n开始重命名子目录并更新路径...")
    rename_subdirectories_and_update_paths(args.nerfstudio_root, frames)
    
    # 保存更新后的 transforms.json
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已更新 transforms.json: {transforms_path}")

    output_root = args.output_root
    images_dst = output_root / "images"
    sparse_text = output_root / "sparse" / "0"
    sparse_text.mkdir(parents=True, exist_ok=True)

    ensure_images(images_src, images_dst, args.copy_images)
    camera_mapping = group_cameras(frames)
    write_cameras_txt(sparse_text / "cameras.txt", camera_mapping)
    write_images_txt(sparse_text / "images.txt", frames, camera_mapping)
    write_points3d_txt(sparse_text / "points3D.txt")

    print(f"已生成 COLMAP 文本模型: {sparse_text}")
    
    if args.reconstruct:
        # 运行 COLMAP 重建流程
        database_path = output_root / "database.db"
        # 删除已存在的数据库（如果存在）
        if database_path.exists():
            database_path.unlink()
        
        print("\n开始 COLMAP 重建流程...")
        # 1. 特征提取
        print("\n[1/4] 特征提取...")
        run_feature_extraction(
            args.colmap_bin, database_path, images_dst, args.use_gpu
        )
        
        # 2. 特征匹配
        print("\n[2/4] 特征匹配...")
        run_feature_matching(args.colmap_bin, database_path, args.use_gpu)
        
        # 2.5. 更新 images.txt 以匹配数据库中的图像名称
        print("\n[2.5/4] 同步图像名称...")
        update_images_txt_from_database(database_path, sparse_text / "images.txt")
        
        # 3. 点三角化
        print("\n[3/4] 点三角化...")
        sparse_triangulated = output_root / "sparse" / "0_triangulated"
        run_point_triangulator(
            args.colmap_bin,
            database_path,
            images_dst,
            sparse_text,
            sparse_triangulated,
        )
        sparse_triangulated_txt = output_root / "sparse" / "0_triangulated_txt"
        convert_model(
            args.colmap_bin, sparse_triangulated, sparse_triangulated_txt, "TXT"
        )
        
        # 4. 导出 PLY
        print("\n[4/4] 导出稀疏点云...")
        ply_output = output_root / "sparse_pcd.ply"
        export_ply(args.colmap_bin, sparse_triangulated, ply_output)
        print(f"\n稀疏点云已保存到: {ply_output}")
        
        # 将 PLY 文件复制到 nerfstudio-root 路径下
        ply_dst = args.nerfstudio_root / "sparse_pcd.ply"
        if ply_output.exists():
            shutil.copy2(ply_output, ply_dst)
            print(f"已复制稀疏点云到: {ply_dst}")
        else:
            print(f"警告: PLY 文件不存在，无法复制: {ply_output}")
        
        # 将三角化后的模型复制回 sparse/0（可选）
        print("\n更新 sparse/0 目录...")
        for file in ["cameras.txt", "images.txt", "points3D.txt"]:
            src = sparse_triangulated_txt / file
            dst = sparse_text / file
            if src.exists():
                shutil.copy2(src, dst)
        print(f"已更新 COLMAP 文本模型: {sparse_text}")
    
    if args.export_binary:
        sparse_bin = output_root / "sparse_bin" / "0"
        export_binary_model(args.colmap_bin, sparse_text, sparse_bin)
        print(f"Binary 模型已写入: {sparse_bin}")


if __name__ == "__main__":
    main()

