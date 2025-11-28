#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据提供的帧编号列表，过滤一个 NeRFstudio 的 transforms.json 文件。
新增功能:
1. 将 file_path 重命名为 "images/Cam{number:03d}.jpg"。
2. 检查 w 和 h, 如果不是 1024x1024, 则缩放 w, h 及内参 (fl_x, fl_y, cx, cy)。
3. 根据帧编号对输出的 frames 列表进行从小到大排序。
4. 添加一个从 "00" 开始递增的字符串 "camera_label" 字段。
"""

import argparse
import json
import os
import re

def filter_transforms_json(input_path, output_path, frame_numbers_list):
    """
    读取、过滤、重命名、缩放并排序
    """
    print(f"正在读取输入的 JSON 文件: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在 {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 JSON 文件: {e}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    try:
        frame_numbers_set = set(int(n) for n in frame_numbers_list)
        print(f"将要保留的帧编号: {frame_numbers_set}")
    except ValueError:
        print("错误: 列表中的编号必须是数字。")
        return

    # 编译一个正则表达式来从 "images/frame_00156.jpg" 中提取数字 "156"
    pattern = re.compile(r"images/frame_(\d+)\.jpg")

    original_frames = data.get("frames", [])
    filtered_frames = []
    
    target_w = 1024
    target_h = 1024

    for frame in original_frames:
        file_path = frame.get("file_path", "")
        match = pattern.search(file_path)
        
        if match:
            frame_num = int(match.group(1)) -1
            
            if frame_num in frame_numbers_set:
                new_frame = frame.copy()
                
                # --- 1. 检查和缩放内参 ---
                w_old = new_frame.get("w")
                h_old = new_frame.get("h")
                fl_x_old = new_frame.get("fl_x")
                fl_y_old = new_frame.get("fl_y")
                cx_old = new_frame.get("cx")
                cy_old = new_frame.get("cy")

                # 检查是否所有必要的内参都存在
                if any(v is None for v in [w_old, h_old, fl_x_old, fl_y_old, cx_old, cy_old]):
                    print(f"  [错误] {file_path} 缺少必要的内参 (w, h, fl_x, fl_y, cx, cy)。跳过此帧。")
                    continue
                
                if w_old == target_w and h_old == target_h:
                    print(f"  [信息] {file_path} 已经是 {target_w}x{target_h}。")
                else:
                    print(f"  [缩放] {file_path} 从 ({w_old}x{h_old}) 缩放到 ({target_w}x{target_h})...")
                    # 检查 w_old 或 h_old 是否为 0 以避免除零错误
                    if w_old == 0 or h_old == 0:
                        print(f"  [错误] {file_path} 的 w 或 h 为 0。无法缩放。跳过此帧。")
                        continue
                        
                    scale_x = target_w / w_old
                    scale_y = target_h / h_old
                    
                    new_frame["w"] = target_w
                    new_frame["h"] = target_h
                    new_frame["fl_x"] = fl_x_old * scale_x
                    new_frame["fl_y"] = fl_y_old * scale_y
                    new_frame["cx"] = cx_old * scale_x
                    new_frame["cy"] = cy_old * scale_y
                
                # --- 2. 重命名 file_path (已修改格式) ---
                # 使用 :03d 格式来确保数字至少为三位数，不足则补零
                new_frame["file_path"] = f"images/{frame_num:02d}/000000.webp"
                new_frame["camera_label"] = f"{frame_num:02d}"
                filtered_frames.append(new_frame)
                print(f"  [处理完成] {file_path} -> {new_frame['file_path']}")
            
            else:
                print(f"  [跳过] {file_path} (编号 {frame_num} 不在列表中)")
        else:
            print(f"  [跳过] {file_path} (格式不匹配)")

    output_data = {"ply_file_path": "sparse_pcd.ply","frames": filtered_frames}
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"成功保存处理后的 JSON 文件到: {os.path.abspath(output_path)}")
    except IOError as e:
        print(f"保存文件时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="根据帧编号列表过滤 transforms.json 文件。")
    
    parser.add_argument("--input_json", type=str,default="/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/colmap/nerfstudio_project/transforms.json",
                        help="输入的 transforms.json 文件路径")
    
    parser.add_argument("--output_json", type=str, default="/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio/transforms.json",
                        help="输出的新的 .json 文件路径")
    
    parser.add_argument("--list", type=int, nargs='+', default=[1],
                        help="要保留的帧编号列表 (例如: --list 156 78 100)")
    
    args = parser.parse_args()
    

    args.list = range(17)
    filter_transforms_json(args.input_json, args.output_json, args.list)

if __name__ == "__main__":
    main()