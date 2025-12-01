import os
import shutil
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from PIL import Image
import sqlite3
import yaml

import sqlite3
import os


def save_colmap_db_stats_to_txt(db_path, output_txt_path):
    """
    读取 COLMAP database.db 并将详细的匹配统计信息保存到指定的 txt 文件中。
    """
    if not os.path.exists(db_path):
        error_msg = f"Error: Database not found at {db_path}"
        print(error_msg)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
             f.write(error_msg)
        return

    print(f"Saving matching statistics to {output_txt_path}...")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. 获取所有图片的 ID 和名称
        cursor.execute("SELECT image_id, name FROM images ORDER BY name;")
        # 使用字典存储，方便后续查找： {image_id: image_name}
        images = {row[0]: row[1] for row in cursor.fetchall()}
        total_images = len(images)

        # 2. 统计匹配信息
        # pair_id = image_id1 * 2147483647 + image_id2
        MAX_IMAGE_ID = 2147483647
        
        # 初始化每个图片的匹配特征点计数器
        matches_per_image = {img_id: 0 for img_id in images.keys()}
        # 初始化每个图片的匹配图像对计数器 (和多少张其他图片产生了关联)
        pairs_per_image = {img_id: 0 for img_id in images.keys()}

        total_matched_pairs = 0
        total_feature_matches = 0

        # 查询 two_view_geometries 表，只统计 rows > 0 的有效匹配
        cursor.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0;")
        for pair_id, rows in cursor.fetchall():
            total_matched_pairs += 1
            total_feature_matches += rows
            
            image_id1 = pair_id // MAX_IMAGE_ID
            image_id2 = pair_id % MAX_IMAGE_ID
            
            # 累加特征点数
            matches_per_image[image_id1] += rows
            matches_per_image[image_id2] += rows
            # 累加关联图像对数
            pairs_per_image[image_id1] += 1
            pairs_per_image[image_id2] += 1

        conn.close()

        # 3. 将结果写入 TXT 文件
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("================ COLMAP Matching Statistics ================\n")
            f.write(f"Database Path: {db_path}\n\n")
            
            f.write("--- Summary ---\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"Total Matched Image Pairs: {total_matched_pairs}\n")
            f.write(f"Total Verified Feature Matches: {total_feature_matches}\n")
            
            avg_matches = total_feature_matches * 2 / total_images if total_images > 0 else 0
            f.write(f"Average Feature Matches per Image: {avg_matches:.1f}\n\n")

            if total_matched_pairs == 0:
                f.write("[WARNING] NO MATCHES FOUND! Reconstruction will likely fail.\n")
            
            f.write("--- Details per Image ---\n")
            f.write(f"{'Image Name':<30} | {'Matched Pairs':<15} | {'Total Feature Matches':<25}\n")
            f.write("-" * 75 + "\n")

            # 按图片名称排序输出
            # 如果你想按匹配数量排序，可以把 sorted 的 key 改为: lambda x: matches_per_image[x[0]]
            for img_id, img_name in sorted(images.items(), key=lambda x: x[1]):
                num_pairs = pairs_per_image.get(img_id, 0)
                num_features = matches_per_image.get(img_id, 0)
                f.write(f"{img_name:<30} | {num_pairs:<15} | {num_features:<25}\n")
            
            f.write("-" * 75 + "\n")
            f.write("End of Report\n")

        print("Statistics saved successfully.")

    except Exception as e:
        error_msg = f"An error occurred while reading the database: {e}"
        print(error_msg)
        # 尝试把错误信息也写进文件
        try:
            with open(output_txt_path, 'a', encoding='utf-8') as f:
                f.write(f"\n[ERROR] {error_msg}\n")
        except:
            pass

def init_parser():
    parser = ArgumentParser("COLMAP data setup and sparse reconstruction arguments.")
    parser.add_argument("--subject", "-s", default='Actor01', type=str, help="Subject folder name.")
    parser.add_argument("--sequence", "-q", default='Sequence1', type=str, help="Sequence folder name.")
    parser.add_argument("--resolution", "-r", default='4x', type=str, help="Resolution folder.")
    parser.add_argument("--triangulator","-t",default=True,type=bool )
    parser.add_argument("--ff", default=460, type=int, help="First frame number.")
    parser.add_argument("--no_gpu",type=bool, default=False)
    parser.add_argument("--is_ba",type=bool, default=True)
    parser.add_argument("--n", default=5, type=int, help="Number of first elements to get indices for.")
    parser.add_argument(
        "--view_config",
        default=Path("/mnt/cvda/cvda_phava/code/Han/PhyAvatar/config/Diffuman/Hq_4_2_DNA_48.py"),
        type=Path,
        help="包含 front_view/side_view/back_view/pre_view 定义的配置文件路径。",
    )
    
    return parser


def load_view_config(config_path: Path):
    """
    从 YAML 或 Python 配置文件中加载 front/side/back/pre 视角列表与附加参数。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # Load from Python or YAML file
    if config_path.suffix == '.py':
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            raw_cfg = getattr(config_module, 'config', {})
        except Exception as exc:
            raise RuntimeError(f"加载 Python 配置失败: {exc}") from exc
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            raise RuntimeError(f"加载 YAML 配置失败: {exc}") from exc

    if not raw_cfg:
        raise ValueError(f"配置文件为空或格式错误: {config_path}")

    # Extract view config
    required_views = ["front_view", "side_view", "back_view", "pre_view"]
    config = {}
    for key in required_views:
        values = raw_cfg.get(key)
        if not isinstance(values, list):
            raise ValueError(f"配置文件中缺少或格式错误: {key} (应为列表)")
        config[key] = values

    # Extract colmap_construction.py args (fallback to legacy key)
    args_obj = raw_cfg.get("colmap_construction.py") or raw_cfg.get("args_config")
    if not isinstance(args_obj, dict):
        raise ValueError("配置文件中缺少或格式错误: colmap_construction.py / args_config (应为字典)")
    args_override = args_obj.copy()

    file_type = "Python" if config_path.suffix == '.py' else "YAML"
    print(f"已从 {config_path} 加载 {file_type} 视角配置及参数。")
    return config, args_override

def get_base_data_path(args: argparse.Namespace) -> Path:
    # Support both HQ_data_root (new) and AHQ_data_root (legacy)
    root = getattr(args, "HQ_data_root", None) or getattr(args, "AHQ_data_root", None)
    if root is None:
        raise ValueError("缺少必需参数: HQ_data_root 或 AHQ_data_root")
    if not hasattr(args, "subject") or not args.subject:
        raise ValueError("缺少必需参数: subject")
    if not hasattr(args, "sequence") or not args.sequence:
        raise ValueError("缺少必需参数: sequence")
    if not hasattr(args, "resolution") or not args.resolution:
        raise ValueError("缺少必需参数: resolution")
    return Path(root) / args.subject / args.sequence / args.resolution


def get_colmap_root(args: argparse.Namespace) -> Path:
    configured = getattr(args, "colmap_root", None)
    if configured:
        return Path(configured)
    return get_base_data_path(args) / "colmap"


def prepare_gs2mesh_data_folder(args):
    """
    准备 gs2mesh 的数据文件夹。
    """
    _root = get_colmap_root(args)
    if _root.exists(): 
        delete = input("输出路径已存在。是否授权删除当前文件夹？ (Y/N) ")
        if delete.lower() == 'y':
            (_root / 'database.db').unlink(missing_ok=True)
            shutil.rmtree(_root)
        else:
            print("\033[91m输出路径已存在，这可能会导致 COLMAP 出错。\033[0m")
    _root.mkdir(parents=True, exist_ok=True)
    _txt = _root / 'txt'
    _txt.mkdir(parents=True, exist_ok=True)
    # 原始 images 文件夹（保留，以防后续需要）
    (_root / 'images').mkdir(parents=True, exist_ok=True)
    # 供 COLMAP 使用的倒置图片文件夹
    _images = _root / 'invert_images'
    _images.mkdir(parents=True, exist_ok=True)
    print(f"\n为 gs2mesh 创建数据文件夹于: {_root}\n")
    return _root, _txt, _images

# MODIFIED: Added 'img_list' argument
def export_first_frames(args, _images, img_list):
    """
    从每个相机复制第一帧图像（无任何额外处理）。
    """
    images_root = get_base_data_path(args) / 'rgbs'

    print(f"正在从指定的相机列表直接复制第一帧到 {_images}")
    for cid in tqdm(img_list):
        ff_src = images_root / cid / f'{cid}_rgb{args.ff:06d}.jpg'
        ff_dest = _images / f'{cid}.jpg'
        if not ff_src.exists():
            print(f"\n警告: 源文件 {ff_src} 不存在，跳过。")
            continue
        try:
            shutil.copy(ff_src, ff_dest)
        except Exception as e:
            print(f"\n警告: 复制 {ff_src} 失败（{e}）")
    print(f"第一帧复制完成。\n")

def convert_local_to_colmap(R_local, t_local):
    """
    将相机外参从 camera-to-world (局部) 转换为 world-to-camera (全局) 坐标系。
    """
    R_global = R_local.T
    t_global = -R_local.T @ t_local
    return R_global, t_global

# MODIFIED: Added 'img_list' argument
def generate_final_colmap_files(db_images_path, calibration_csv_path, output_dir, img_list):
    """
    读取标定数据和COLMAP数据库的图像列表，以生成最终的、正确排序的
    cameras.txt 和 images.txt 文件，用于三角化。
    """
    # 仅使用原始标定数据
    print("--- 正在为三角化生成最终的 COLMAP txt 文件 ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 将标定数据读入一个字典以便快速查找
    calib_df = pd.read_csv(calibration_csv_path, index_col=0)
    # NEW: Filter the calibration data to only include cameras from img_list
    calib_df = calib_df.loc[calib_df.index.isin(img_list)]
    calib_data = calib_df.to_dict('index')
    print(f"已加载并筛选了 {len(calib_data)} 个相机的标定数据。")

    # 2. 从数据库导出文件中读取正确的图像顺序
    db_df = pd.read_csv(
        db_images_path, sep=r'\s+', skiprows=3, header=None,
        names=['image_id', 'name', 'camera_id', 'prior_qw', 'prior_qx', 'prior_qy', 'prior_qz', 'prior_tx', 'prior_ty', 'prior_tz']
    )
    # NEW: Filter the database image list to only include images from img_list
    image_filenames = [f"{cam}.jpg" for cam in img_list]
    db_df = db_df[db_df['name'].isin(image_filenames)]
    
    db_df_sorted = db_df.sort_values(by='image_id').reset_index(drop=True)
    print(f"成功读取 {len(db_df_sorted)} 张图像的正确顺序。")

    camera_lines = ""
    image_lines = ""

    # 3. 按照正确的顺序遍历并生成文件内容
    for index, row in db_df_sorted.iterrows():
        image_name_with_ext = row['name'] # 这是正确的文件名，例如 "Cam160.jpg"
        camera_name = Path(image_name_with_ext).stem
        
        if camera_name not in calib_data:
            print(f"警告: 数据库中的 {camera_name} 在筛选后的标定数据中未找到")
            continue

        calib = calib_data[camera_name]
        
        camera_id = row['camera_id']
        image_id = row['image_id']

        rotvec = np.array([calib['rx'], calib['ry'], calib['rz']])
        translation = np.array([calib['tx'], calib['ty'], calib['tz']])
        focal = np.array([calib['fx'], calib['fy']])
        principal = np.array([calib['px'], calib['py']])
        w, h = int(calib['w']), int(calib['h'])

        # 可选：对指定相机进行宽高反转或测试模式几何/内参修改
        try:
            cam_idx = int(camera_name.replace("Cam", ""))
        except ValueError:
            cam_idx = None

        # 使用原始标定参数
        focal *= np.array([w, h])
        principal *= np.array([w, h])

        # 将 camera-to-world 转换为 world-to-camera（COLMAP 格式）
        rotmat = R.from_rotvec(rotvec).as_matrix()
        rotmat, translation = convert_local_to_colmap(rotmat, translation)
        
        quaternion = R.from_matrix(rotmat).as_quat()
        quaternion = np.roll(quaternion, 1)

        image_lines += f"{image_id} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]} {translation[0]} {translation[1]} {translation[2]} {camera_id} {image_name_with_ext}\n\n"
        camera_lines += f"{camera_id} PINHOLE {w} {h} {focal[0]} {focal[1]} {principal[0]} {principal[1]}\n"
    
    # 将文件写入输出目录 (例如, txt/1)
    with open(output_dir / "cameras.txt", "w") as f: f.write(camera_lines)
    with open(output_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[]\n")
        f.write(image_lines)
    with open(output_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    
    print(f"成功在 {output_dir} 中生成最终的 COLMAP 文件\n")

# MODIFIED: Added 'img_list' argument
def run_colmap(_root, _txt, _images, args, img_list):
    """
    运行 COLMAP 工作流
    """
    _sparse = _root / 'sparse'
    _dense = _root / 'dense'
    use_gpu = int(not args.no_gpu)

    # NEW: Create a text file listing the specific images to process
    image_list_path = _txt / 'image_list.txt'
    with open(image_list_path, 'w') as f:
        for cam_name in img_list:
            f.write(f"{cam_name}.jpg\n")
    print(f"已创建图像列表文件于: {image_list_path}")

    extract_log_path = os.path.join(_txt, "feature_extraction.log")
    match_log_path = os.path.join(_txt, "feature_matching.log")
    stats_output_path = os.path.join(_txt, "matching_stats.txt")
    BA_output_path = os.path.join(_txt, "comparison_report.txt")
    
    ## 特征提取
    # MODIFIED: Use --image_list_path to restrict feature extraction
    feat_extracton_cmd = (
        f"colmap feature_extractor "
        f"--database_path {_root}/database.db "
        f"--image_path {_images} "
        f"--image_list_path {image_list_path} "
        f"--SiftExtraction.peak_threshold 0.005 "
        f"--ImageReader.single_camera 0 "
        f"--SiftExtraction.edge_threshold 80 "
        f"--SiftExtraction.use_gpu {use_gpu} "
        f" --SiftExtraction.first_octave -1 "
        f" --SiftExtraction.estimate_affine_shape=true "
        f" --SiftExtraction.domain_size_pooling=true "
        f" > {extract_log_path} 2>&1"
    )
    if os.system(feat_extracton_cmd) != 0: logging.error("特征提取失败。"); exit(1)

    ## 特征匹配
    feat_matching_cmd = (
        f"colmap exhaustive_matcher "
        f"--database_path {_root}/database.db "
        f"--SiftMatching.use_gpu {use_gpu} "
        f"--SiftMatching.min_num_inliers 5 "
        f" > {match_log_path} 2>&1"
    )
    if os.system(feat_matching_cmd) != 0: logging.error("特征匹配失败。"); exit(1)

    _sparse.mkdir(parents=True, exist_ok=True)

    save_colmap_db_stats_to_txt(f"{_root}/database.db", stats_output_path)

    if args.triangulator:
        ## 新流程：在匹配后，生成最终的、排序正确的txt文件
        db_images_txt_path = _txt / 'db_images.txt'
        calibration_csv_path = get_base_data_path(args) / 'calibration.csv'
        final_txt_dir = _txt / '1'

        # 导出数据库中的图像表，以获取正确的顺序
        export_table_to_txt(f'{_root}/database.db', 'images', db_images_txt_path)
        # MODIFIED: Pass img_list to the file generation function
        generate_final_colmap_files(db_images_txt_path, calibration_csv_path, final_txt_dir, img_list)
        # 根据数据库更新 images.txt 中的图像名称，确保与数据库一致
        images_txt_path = final_txt_dir / 'images.txt'
        update_images_txt_from_database(f'{_root}/database.db', images_txt_path)
        print("start triangulator.....")
        triangulate_cmd = f"colmap point_triangulator --database_path {_root}/database.db --image_path {_images} --input_path {final_txt_dir} --output_path {_sparse}"
        if os.system(triangulate_cmd) != 0: logging.error("点云三角化失败。"); exit(1)
    else :
        print("start mapper.....")
        mapper_cmd = f"colmap mapper --database_path {_root}/database.db --image_path {_images} --output_path {_sparse} --Mapper.tri_ignore_two_view_tracks 0 --Mapper.ba_refine_focal_length 0  --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 --Mapper.init_min_num_inliers 3 "
        if os.system(mapper_cmd) != 0: 
            print("稀疏重建失败。")
            logging.error("稀疏重建失败。"); exit(1)


    if args.is_ba:
        _sparse_ba = _sparse / "_ba" 
        _sparse_ba.mkdir(exist_ok=True)
        # 1. 记录初始状态
        os.system(f"echo '--- Before BA ---' >> {BA_output_path}")
        os.system(f"colmap model_analyzer --path {_sparse} >> {BA_output_path}")

        # 2. 运行 BA (输出到新路径，并把日志也重定向到文件以便检查)
        # 增加 --BundleAdjustment.refine_... 参数强制开启某些优化，看看是否有变化
        print("Starting Bundle Adjustment...")
        ba_log_path = BA_output_path.replace(".txt", "_log.txt")
        cmd = f"colmap bundle_adjuster --input_path {_sparse} --output_path {_sparse_ba} --BundleAdjustment.refine_principal_point 1 >> {ba_log_path} 2>&1"
        exit_code = os.system(cmd)

        if exit_code != 0:
            print(f"Error: Bundle Adjustment failed with exit code {exit_code}. Check {ba_log_path}")
        else:
            print("Bundle Adjustment finished.")
            # 3. 记录最终状态 (分析新路径下的模型)
            os.system(f"echo '\n--- After BA ---' >> {BA_output_path}")
            # 确保新路径存在才分析
            if os.path.exists(_sparse_ba):
                os.system(f"colmap model_analyzer --path {_sparse_ba} >> {BA_output_path}")
            else:
                os.system(f"echo 'Error: Output model not found at {_sparse_ba}' >> {BA_output_path}")

    _dense.mkdir(parents=True, exist_ok=True)
    os.system(f"colmap image_undistorter --image_path {_images} --input_path {_sparse} --output_path {_dense}")
    os.system(f"colmap model_converter --input_path {_dense}/sparse --output_path {_dense}/sparse --output_type TXT")
    os.system(f"colmap model_converter --input_path {_dense}/sparse --output_path {_dense}/sparse/3D.ply --output_type PLY")
    

    # shutil.rmtree(_sparse)
    # _sparse.mkdir(exist_ok=True)
    (_sparse / '0').mkdir(exist_ok=True)
    os.system(f"mv {_dense}/sparse/* {_sparse}/0")
    shutil.rmtree(_dense)

    # 额外导出一个整体点云文件 spare_pcd.ply 到 COLMAP 根目录下，便于后续可视化/使用
    spare_pcd_path = _root / 'spare_pcd.ply'
    os.system(f"colmap model_converter --input_path {_sparse}/0 --output_path {spare_pcd_path} --output_type PLY")
   


def export_table_to_txt(db_path, table_name, output_txt_path):
    """
    将 SQLite 数据库中的表导出为 TXT 文件。
    """
    print(f"正在从 {db_path} 导出 '{table_name}' 表到 {output_txt_path}")
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件 '{db_path}' 未找到")
        return
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        column_names = [info[1] for info in cursor.fetchall()]
        header = "\t".join(column_names)
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"# Table: {table_name}\n# Columns: {header}\n# -----------------------------------\n")
            for row in rows:
                f.write("\t".join(map(str, row)) + '\n')
        print(f"成功导出 {len(rows)} 行到 '{output_txt_path}'.")
    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
    finally:
        if conn:
            conn.close()

def update_images_txt_from_database(database_path, images_txt_path):
    """
    从数据库中读取图像名称，并更新 images.txt 文件，确保图像名称一致。
    
    数据库中的图像名称是 feature_extractor 根据实际文件系统生成的，
    必须与 images.txt 中的名称完全匹配，否则 point_triangulator 会失败。
    """
    print(f"正在根据数据库更新 images.txt: {images_txt_path}")
    
    # 从数据库读取图像名称（按 image_id 排序）
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_id, name FROM images ORDER BY image_id"
    )
    db_images = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    
    if not db_images:
        raise ValueError("数据库中未找到图像记录")
    
    # 读取现有的 images.txt
    with open(images_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 解析并更新图像名称
    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 保留注释行和空行
        if not line or line.startswith("#"):
            updated_lines.append(lines[i])
            i += 1
            continue
        
        # 解析图像行
        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            # 获取数据库中的图像名称
            if image_id in db_images:
                db_name = db_images[image_id]
                # 替换图像名称（保留其他所有字段）
                new_line = (
                    f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} "
                    f"{parts[5]} {parts[6]} {parts[7]} {parts[8]} {db_name}\n"
                )
                updated_lines.append(new_line)
            else:
                print(f"警告: 图像 ID {image_id} 在数据库中未找到，保留原名称")
                updated_lines.append(lines[i])
        else:
            updated_lines.append(lines[i])
        
        i += 1
        # 跳过下一行（POINTS2D 行，通常是空的）
        if i < len(lines):
            updated_lines.append(lines[i])
            i += 1
    
    # 写回文件
    with open(images_txt_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)
    
    print(f"已更新 images.txt，使用数据库中的图像名称（共 {len(db_images)} 个图像）")

def main():
    parser = init_parser()
    args = parser.parse_args()

    view_cfg, args_override = load_view_config(Path(args.view_config))
    hq_override = args_override.pop("HQ_data_root", None)
    if hq_override is not None:
        setattr(args, "HQ_data_root", Path(hq_override))
    elif ahq_override is not None:
        setattr(args, "HQ_data_root", Path(ahq_override))  # Map legacy to new name

    if "colmap_root" in args_override and args_override["colmap_root"] is not None:
        args_override["colmap_root"] = Path(args_override["colmap_root"])
    # colmap_root will be computed from other args if not provided

    for key, value in args_override.items():
        setattr(args, key, value)

    front_view = view_cfg["front_view"]
    side_view = view_cfg["side_view"]
    back_view = view_cfg["back_view"]
    pre_view = view_cfg["pre_view"]

    view_list = front_view + side_view + back_view 

    img_list = [f"Cam{i:03d}" for i in view_list]
    _root, _txt, _images = prepare_gs2mesh_data_folder(args)

    _images = _root / 'images'
    print(f"[模式] 使用 images 路径: {_images}")

    if pre_view:
        sorted_view_list = sorted(view_list)
        # 获取每个元素在排序后的索引
        index_value_pairs = []
        for val in pre_view:
            if val in sorted_view_list:
                idx = sorted_view_list.index(val) 
                index_value_pairs.append(f"{idx:02d} - {val}")
            else:
                print(f"警告: 预设视角 {val} 不在 view_list 中，已跳过")
        
        # 保存到 _root/index.txt
        index_file = _root / f"index.txt"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(index_value_pairs))
        print(f"\n已保存索引到: {index_file}")
        print(f"预设视角: {pre_view}")
        print(f"排序后的 view_list: {sorted_view_list}")
        print(f"对应的索引-值: {index_value_pairs}")
    
    # MODIFIED: Pass the img_list to the function
    export_first_frames(args, _images, img_list)
    
    # MODIFIED: Pass the img_list to the main COLMAP function
    run_colmap(_root, _txt, _images, args, img_list)

if __name__ == "__main__":
    main()