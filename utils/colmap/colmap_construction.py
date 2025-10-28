import os
import shutil
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
import sqlite3
from defaults import DEFAULTS

gs2mesh_path = Path.cwd() / 'gs2mesh'

def init_parser():
    parser = ArgumentParser("COLMAP data setup and sparse reconstruction arguments.")
    parser.add_argument("--subject", "-s", default='Actor01', type=str, help="Subject folder name.")
    parser.add_argument("--sequence", "-q", default='Sequence1', type=str, help="Sequence folder name.")
    parser.add_argument("--resolution", "-r", default='4x', type=str, help="Resolution folder.")
    parser.add_argument("--ff", default=460, type=int, help="First frame number.")
    parser.add_argument("--no_gpu", action='store_true')
    return parser

def prepare_gs2mesh_data_folder(args):
    """
    准备 gs2mesh 的数据文件夹。
    """
    _root = gs2mesh_path / 'data' / 'custom' / f'{args.subject}_{args.sequence}'
    if _root.exists(): 
        delete = input("输出路径已存在。是否授权删除当前文件夹？ (Y/N) ")
        if delete.lower() == 'y':
            shutil.rmtree(_root)
        else:
            print("\033[91m输出路径已存在，这可能会导致 COLMAP 出错。\033[0m")
    _root.mkdir(parents=True, exist_ok=True)
    _txt = _root / 'txt'
    _txt.mkdir(parents=True, exist_ok=True)
    _images = _root / 'images'
    _images.mkdir(parents=True, exist_ok=True)
    print(f"\n为 gs2mesh 创建数据文件夹于: {_root}\n")
    return _root, _txt, _images

# MODIFIED: Added 'img_list' argument
def export_first_frames(args, _images, img_list):
    """
    从每个相机复制第一帧图像。
    """
    images_root = DEFAULTS['AHQ_data_root'] / args.subject / args.sequence / args.resolution / 'rgbs'
    print(f"正在从指定的相机列表复制第一帧到 {_images}")
    # MODIFIED: Iterate through the provided img_list instead of all directories
    for cid in tqdm(img_list):
        ff_src = images_root / cid / f'{cid}_rgb{args.ff:06d}.jpg'
        ff_dest = _images / f'{cid}.jpg'
        if not ff_src.exists():
            print(f"\n警告: 源文件 {ff_src} 不存在，跳过。")
            continue
        shutil.copy(ff_src, ff_dest)
    print(f"第一帧复制成功。\n")

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
        focal *= np.array([w, h])
        principal *= np.array([w, h])

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

    ## 特征提取
    # MODIFIED: Use --image_list_path to restrict feature extraction
    feat_extracton_cmd = (
        f"colmap feature_extractor "
        f"--database_path {_root}/database.db "
        f"--image_path {_images} "
        f"--image_list_path {image_list_path} " # This is the key change
        f"--SiftExtraction.peak_threshold 0.003 "
        f"--ImageReader.single_camera 0 "
        f"--SiftExtraction.edge_threshold 80 "
        f"--SiftExtraction.use_gpu {use_gpu}"
    )
    if os.system(feat_extracton_cmd) != 0: logging.error("特征提取失败。"); exit(1)

    ## 特征匹配
    feat_matching_cmd = f"colmap exhaustive_matcher --database_path {_root}/database.db --SiftMatching.use_gpu {use_gpu} --ExhaustiveMatching.block_size 200 --SiftMatching.max_num_matches 32768"
    if os.system(feat_matching_cmd) != 0: logging.error("特征匹配失败。"); exit(1)

    ## 新流程：在匹配后，生成最终的、排序正确的txt文件
    db_images_txt_path = _txt / 'db_images.txt'
    calibration_csv_path = DEFAULTS['AHQ_data_root'] / args.subject / args.sequence / args.resolution / 'calibration.csv'
    final_txt_dir = _txt / '1'
    
    # 导出数据库中的图像表，以获取正确的顺序
    export_table_to_txt(f'{_root}/database.db', 'images', db_images_txt_path)
    # MODIFIED: Pass img_list to the file generation function
    generate_final_colmap_files(db_images_txt_path, calibration_csv_path, final_txt_dir, img_list)

    ## 点云三角化
    _sparse.mkdir(parents=True, exist_ok=True)
    triangulate_cmd = f"colmap point_triangulator --database_path {_root}/database.db --image_path {_images} --input_path {final_txt_dir} --output_path {_sparse}"
    if os.system(triangulate_cmd) != 0: logging.error("点云三角化失败。"); exit(1)
    
    ## 图像去畸变和模型转换
    _dense.mkdir(parents=True, exist_ok=True)
    os.system(f"colmap image_undistorter --image_path {_images} --input_path {_sparse} --output_path {_dense}")
    os.system(f"colmap model_converter --input_path {_dense}/sparse --output_path {_dense}/sparse --output_type TXT")
    os.system(f"colmap model_converter --input_path {_dense}/sparse --output_path {_dense}/sparse/3D.ply --output_type PLY")
    
    # 重组文件结构以满足 gs2mesh 的要求
    shutil.rmtree(_sparse)
    _sparse.mkdir(exist_ok=True)
    (_sparse / '0').mkdir(exist_ok=True)
    os.system(f"mv {_dense}/sparse/* {_sparse}/0")
    shutil.rmtree(_dense)
    (_root / 'database.db').unlink(missing_ok=True)


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

def main():
    parser = init_parser()
    args = parser.parse_args()

    # NEW: Define the specific list of cameras to process
    # img_list = [
    #     'Cam006', 'Cam008', 'Cam021', 'Cam023', 'Cam110', 'Cam112', 'Cam126', 
    #     'Cam127', 'Cam128', 'Cam135', 'Cam136', 'Cam143', 'Cam144', 'Cam151', 
    #     'Cam152', 'Cam159', 'Cam160'
    # ]
    img_list = [
    'Cam005', 'Cam007', 'Cam021', 'Cam023', 'Cam037', 'Cam039', 'Cam053', 
    'Cam055', 'Cam077', 'Cam079', 'Cam094', 'Cam095', 'Cam110', 'Cam112', 
    'Cam126', 'Cam127', 'Cam128', 'Cam135', 'Cam136', 'Cam143', 'Cam144', 
    'Cam151', 'Cam152', 'Cam159', 'Cam160', 
]

    _root, _txt, _images = prepare_gs2mesh_data_folder(args)
    
    # MODIFIED: Pass the img_list to the function
    export_first_frames(args, _images, img_list)
    
    # MODIFIED: Pass the img_list to the main COLMAP function
    run_colmap(_root, _txt, _images, args, img_list)

if __name__ == "__main__":
    main()