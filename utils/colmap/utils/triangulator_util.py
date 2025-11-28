"""Due to disordered image and camera IDs causing triangulation failures, I need to update the pre-defined TXT files using the database records."""
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

def convert_local_to_colmap(R_local, t_local):
    """
    将相机外参从 camera-to-world (局部) 转换为 world-to-camera (全局) 坐标系。
    """
    R_global = R_local.T
    t_global = -R_local.T @ t_local
    return R_global, t_global

def triangulation_updata_imagesID(database_path,csv_path=None,img_list=None):

    '''
        return  
            generate_file_dir is --input_path
    '''

    if img_list:
        img_list = [f"Cam{i+1:03d}" for i in range(160)]
    if csv_path:
        csv_path = "/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/calibration.csv"

    db_images_txt_path = database_path/"db_images.txt"
    generate_file_dir = database_path / '1'

    export_table_to_txt(f'{database_path}/database.db', 'images', db_images_txt_path)

    generate_final_colmap_files(db_images_txt_path, csv_path, generate_file_dir, img_list)

    return generate_file_dir