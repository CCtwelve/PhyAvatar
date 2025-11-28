import cv2
import numpy as np
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def setup_output_dirs(base_path):
    """
    创建或清空用于存放无畸变数据的新目录。
    """
    output_img_dir = base_path / "undistorted_images"
    output_mask_dir = base_path / "undistorted_fmasks"

    for dir_path in [output_img_dir, output_mask_dir]:
        if dir_path.exists():
            print(f"警告：正在删除已存在的目录: {dir_path}")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        print(f"已创建目录: {dir_path}")

    return output_img_dir, output_mask_dir


# [!!!] 用这个新版本替换你的 process_dataset 函数 [!!!]
def process_dataset(data_dir):
    """
    对整个数据集执行去畸变处理。
    
    1. 修改 transforms.json
    2. 对 /images/ 中所有图像去畸变
    3. 对 /fmasks/ 中所有掩码去畸变
    """
    base_path = Path(data_dir)
    json_path = base_path / "transforms.json"
    img_dir = base_path / "images"
    mask_dir = base_path / "fmasks"

    if not json_path.exists():
        print(f"错误：在 {json_path} 未找到 transforms.json")
        return
        
    print(f"正在加载原始 JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 创建输出目录
    output_img_dir, output_mask_dir = setup_output_dirs(base_path)
    output_json_path = base_path / "transforms_undistorted.json"
    
    # 2. 准备新的 JSON 数据
    new_data = data.copy()
    new_data['frames'] = [] # 清空 frames，我们将填充新数据
    
    print(f"共找到 {len(data['frames'])} 个相机 (视图) 需要处理...")

    # 3. 遍历每个相机 (frame)
    for frame in tqdm(data['frames'], desc="正在处理相机..."):
        
        # --- a. 处理相机参数 (JSON) ---
        cam_label = frame['camera_label']
        w, h = frame['w'], frame['h']
        
        fl_x, fl_y = frame['fl_x'], frame['fl_y']
        cx, cy = frame['cx'], frame['cy']
        K_orig = np.array([
            [fl_x, 0.0,  cx],
            [0.0,  fl_y, cy],
            [0.0,  0.0,  1.0]
        ], dtype=np.float32)

        # 构建 8 元素 D 向量
        k1 = frame.get('k1', 0.0)
        k2 = frame.get('k2', 0.0)
        p1 = frame.get('p1', 0.0)
        p2 = frame.get('p2', 0.0)
        k3 = frame.get('k3', 0.0)
        k4 = frame.get('k4', 0.0)
        k5 = frame.get('k5', 0.0) 
        k6 = frame.get('k6', 0.0)
        D = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float32)
        
        new_K, roi = cv2.getOptimalNewCameraMatrix(K_orig, D, (w, h), 1.0, (w, h))
        
        # --- [!!! 修复代码 !!!] ---
        # 1. 计算一次映射图 (Map)
        # 我们为每个相机计算一次 (w,h) 映射图
        # map1, map2 是用于 cv2.remap 的
        map1, map2 = cv2.initUndistortRectifyMap(
            K_orig, D, None, new_K, (w, h), cv2.CV_32FC1
        )
        # --- [!!! 修复结束 !!!] ---

        # 准备新的 frame entry
        new_frame = frame.copy()
        # --- [!!! 修复代码 !!!] ---
        # 将 np.float32/np.float64 显式转换为 Python float
        new_frame['fl_x'] = float(new_K[0, 0])
        new_frame['fl_y'] = float(new_K[1, 1])
        new_frame['cx'] = float(new_K[0, 2])
        new_frame['cy'] = float(new_K[1, 2])
        # --- [!!! 修复结束 !!!] ---
        new_frame['camera_model'] = "PINHOLE"
        
        keys_to_remove = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2']
        for key in keys_to_remove:
            new_frame.pop(key, None)
            
        new_data['frames'].append(new_frame)
        
        # --- b. 处理该相机对应的所有图像和掩码 ---
        cam_img_dir_orig = img_dir / cam_label
        cam_mask_dir_orig = mask_dir / cam_label
        
        cam_img_dir_out = output_img_dir / cam_label
        cam_mask_dir_out = output_mask_dir / cam_label
        
        os.makedirs(cam_img_dir_out, exist_ok=True)
        os.makedirs(cam_mask_dir_out, exist_ok=True)
        
        image_files = sorted(list(cam_img_dir_orig.glob("*.jpg")))
        if not image_files:
            print(f"警告：在 {cam_img_dir_orig} 中未找到 {cam_label} 相机的图像")
            continue
            
        for img_path in tqdm(image_files, desc=f"  - Cam {cam_label}", leave=False):
            pose_name_jpg = img_path.name
            pose_name_png = f"{img_path.stem}.png"
            mask_path = cam_mask_dir_orig / pose_name_png
            
            img_distorted = cv2.imread(str(img_path))
            mask_distorted = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            
            if img_distorted is None:
                print(f"警告：跳过，无法加载图像 {img_path}")
                continue
                
            # --- [!!! 修复代码 !!!] ---
            # 2. 使用 cv2.remap 应用映射图
            # 对彩色图像使用线性插值
            img_undistorted = cv2.remap(
                img_distorted, map1, map2, cv2.INTER_LINEAR
            )
            # --- [!!! 修复结束 !!!] ---
            
            cv2.imwrite(str(cam_img_dir_out / pose_name_jpg), img_undistorted)
            
            if mask_distorted is not None:
                # --- [!!! 修复代码 !!!] ---
                # 3. 对掩码使用最近邻插值 (INTER_NEAREST)
                mask_undistorted = cv2.remap(
                    mask_distorted, map1, map2, cv2.INTER_NEAREST
                )
                # --- [!!! 修复结束 !!!] ---
                cv2.imwrite(str(cam_mask_dir_out / pose_name_png), mask_undistorted)
                
    # 4. 保存新的 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    print("\n" + "="*50)
    print("✅ 预处理完成！")
    print(f"新的 JSON 文件已保存到: {output_json_path}")
    print(f"新的图像已保存到: {output_img_dir}")
    print(f"新的掩码已保存到: {output_mask_dir}")
    print("="*50)
    print("\n下一步操作：")
    print("1. 确保你的 Dataset 类已按上一个回答中的说明修改。")
    print("2. 开始训练！")


if __name__ == "__main__":
    process_dataset('/mnt/cvda/cvda_phava/dataset/DNA-Rendering/0023_06/')