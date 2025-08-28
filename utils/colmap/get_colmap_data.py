#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/13 18:23
# @Author  : jc Han
# @help    :

import os
import shutil

# 源路径
source_path = "/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/rgbs/"
# 目标路径
target_path = "/mnt/cvda/cvda_phava/code/Han/3DGS/colmap/data/multipleview/0_15"  # 请替换为你的目标路径
number = '0000'
# 确保目标路径存在
os.makedirs(target_path, exist_ok=True)
cam_list = [22, 37, 38, 55, 78, 92, 94, 109, 125, 126, 127, 134, 151, 158, 159]
# 遍历160个摄像机文件夹
for cam_num in cam_list:
    # 格式化摄像机文件夹名（Cam001到Cam160）
    cam_folder = f"Cam{cam_num:03d}"
    filename = f"{cam_folder}_rgb00{number}.jpg"
    cam_path = os.path.join(source_path, cam_folder,filename)

    copy_path =os.path.join(target_path,cam_folder+".jpg")

    shutil.copy2(cam_path , copy_path)
    print(f"Copied: {cam_path} -> {copy_path}")

print("所有文件处理完成！")