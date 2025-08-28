#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/16 0:22
# @Author  : jc Han
# @help    :

import os
import cv2
from natsort import natsorted  # 按数字顺序排序文件名
import shutil

def images_to_mp4(image_folder, output_path, fps=60, resize=None):
    """
    将文件夹中的jpg图像合成MP4视频

    参数:
        image_folder (str): 包含jpg图像的文件夹路径
        output_path (str): 输出视频路径（如 'output.mp4'）
        fps (int): 帧率（默认30）
        resize (tuple): 可选，目标分辨率 (width, height)
    """

    pic = image_folder+"/Cam127_rgb000460.jpg"
    output_dir = "/mnt/cvda/cvda_phava/code/Han/LHM/train_data/custom_motion/custom_dress/image"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, 100):
        new_name = f"{i+460}.jpg"
        new_path = os.path.join(output_dir, new_name)
        shutil.copy2(pic, new_path)


    images = [img for img in os.listdir(output_dir) if img.endswith(".jpg")]
    images = natsorted(images)
    print(images[0])

    if not images:
        raise ValueError("指定路径下未找到jpg文件！")

    # 读取第一张图像获取尺寸
    sample = cv2.imread(os.path.join(output_dir, images[0]))
    if resize:
        height, width = resize[1], resize[0]  # OpenCV尺寸顺序为 (width, height)
    else:
        height, width = sample.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    shutil.copy2("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/rgbs/Cam127/Cam127_rgb000460.jpg",
                 "/mnt/cvda/cvda_phava/code/Han/LHM/train_data/example_imgs/Cam127.jpg")
    # 逐帧写入视频
    for img_name in images:
        img_path = os.path.join(output_dir, img_name)
        frame = cv2.imread(img_path)
        if resize:
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    # 释放资源
    video.release()
    print(f"视频已生成: {output_path}")


# 使用示例
if __name__ == "__main__":
    image_folder = "/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/rgbs/Cam127"  # 替换为你的图片文件夹路径
    output_path = "/mnt/cvda/cvda_phava/code/Han/LHM/train_data/custom_dress.mp4"  # 输出视频文件名
    images_to_mp4(image_folder, output_path, fps=30)