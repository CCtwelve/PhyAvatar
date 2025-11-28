#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 transforms.json 中的图像尺寸修改为 1024x1024，图像居中放置，多出的部分用黑色填充。
同时修改相机的内参使其符合修改之后的图像。

处理逻辑：
1. 保持原始图像的宽高比
2. 将图像按比例缩放以适应 1024x1024 画布（选择较小的缩放比例以确保图像完全放入）
3. 图像居中放置，多出的部分用黑色填充
4. 调整内参：焦距按比例缩放，主点坐标需要加上偏移量
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def resize_transforms_json(
    input_json_path: Path,
    output_json_path: Path,
    target_w: int = 1024,
    target_h: int = 1024
):
    """
    修改 transforms.json 中的图像尺寸和内参
    
    Args:
        input_json_path: 输入的 transforms.json 路径
        output_json_path: 输出的 transforms.json 路径
        target_w: 目标宽度（默认 1024）
        target_h: 目标高度（默认 1024）
    """
    print(f"读取输入的 JSON 文件: {input_json_path}")
    
    if not input_json_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_json_path}")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    if not frames:
        print("警告: JSON 文件中没有找到 'frames' 键或 frames 为空。")
        return
    
    print(f"找到 {len(frames)} 个相机帧")
    print(f"目标尺寸: {target_w} x {target_h}")
    print()
    
    for idx, frame in enumerate(frames):
        # 获取原始尺寸和内参
        w_old = frame.get('w')
        h_old = frame.get('h')
        fl_x_old = frame.get('fl_x')
        fl_y_old = frame.get('fl_y')
        cx_old = frame.get('cx')
        cy_old = frame.get('cy')
        
        file_path = frame.get('file_path', f'frame_{idx}')
        
        # 检查必要的内参是否存在
        if any(v is None for v in [w_old, h_old, fl_x_old, fl_y_old, cx_old, cy_old]):
            print(f"  [错误] {file_path} 缺少必要的内参。跳过此帧。")
            continue
        
        if w_old == 0 or h_old == 0:
            print(f"  [错误] {file_path} 的 w 或 h 为 0。跳过此帧。")
            continue
        
        # 计算缩放比例（保持宽高比，选择较小的比例以确保图像完全放入）
        scale_x = target_w / w_old
        scale_y = target_h / h_old
        scale = min(scale_x, scale_y)  # 选择较小的缩放比例
        
        # 计算缩放后的图像尺寸
        w_scaled = w_old * scale
        h_scaled = h_old * scale
        
        # 计算居中放置时的偏移量（图像在新画布中的左上角位置）
        offset_x = (target_w - w_scaled) / 2.0
        offset_y = (target_h - h_scaled) / 2.0
        
        # 更新图像尺寸
        frame['w'] = target_w
        frame['h'] = target_h
        
        # 更新内参
        # 焦距按比例缩放
        frame['fl_x'] = fl_x_old * scale
        frame['fl_y'] = fl_y_old * scale
        
        # 主点坐标：先按比例缩放，再加上偏移量
        frame['cx'] = cx_old * scale + offset_x
        frame['cy'] = cy_old * scale + offset_y
        
        print(f"  [{idx+1}/{len(frames)}] {file_path}")
        print(f"    原始尺寸: {w_old} x {h_old}")
        print(f"    缩放比例: {scale:.6f}")
        print(f"    缩放后尺寸: {w_scaled:.2f} x {h_scaled:.2f}")
        print(f"    偏移量: ({offset_x:.2f}, {offset_y:.2f})")
        print(f"    新尺寸: {target_w} x {target_h}")
        print(f"    焦距: ({fl_x_old:.2f}, {fl_y_old:.2f}) -> ({frame['fl_x']:.2f}, {frame['fl_y']:.2f})")
        print(f"    主点: ({cx_old:.2f}, {cy_old:.2f}) -> ({frame['cx']:.2f}, {frame['cy']:.2f})")
        print()
    
    # 保存修改后的 JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"修改后的 transforms.json 已保存到: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将 transforms.json 中的图像尺寸修改为指定尺寸，图像居中放置，多出的部分用黑色填充，并调整内参。'
    )
    parser.add_argument(
        '--input_json',
        type=str,
        default="/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/transforms.json",
        help='输入的 transforms.json 文件路径'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default="/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/transforms_resized.json",
        help='输出的 transforms.json 文件路径'
    )
    parser.add_argument(
        '--target_w',
        type=int,
        default=1024,
        help='目标宽度（默认: 1024）'
    )
    parser.add_argument(
        '--target_h',
        type=int,
        default=1024,
        help='目标高度（默认: 1024）'
    )
    
    args = parser.parse_args()
    
    input_json_path = Path(args.input_json)
    output_json_path = Path(args.output_json)
    
    resize_transforms_json(
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        target_w=args.target_w,
        target_h=args.target_h
    )


if __name__ == '__main__':
    main()

