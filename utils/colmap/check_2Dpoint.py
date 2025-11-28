import json
import cv2
import numpy as np

# --- 1. 定义文件路径 ---
file_list=[6, 8, 21, 23, 110, 112, 126, 127, 128, 135, 136, 143, 144]
for i in file_list:
    # 这是你提供的 JSON 文件路径
    json_path = f'/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio/poses_sapiens/Cam{i:03d}/460.json'

    # ⚠️ 关键：请将这里替换为你的图片路径！
    image_path = f'/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio/images/Cam{i:03d}/460.jpg'  

    # 这是你希望保存的输出图片名称
    output_path = f'/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio/poses_2d_keypoint/Cam{i:03d}_with_keypoints_1024_.jpg'

    # --- 2. 加载数据和图片 ---
    try:
        # 加载 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 提取关键点 (x, y) 坐标
        keypoints = data['instance_info'][0]['keypoints']

        # 使用 OpenCV 读取图片
        img = cv2.imread(image_path)
        
        if img is None:
            raise FileNotFoundError(image_path)
            
        print(f"成功加载图片: {image_path}")
        print(f"成功加载 {len(keypoints)} 个关键点 from {json_path}")

    except FileNotFoundError:
        print(f"错误: 找不到图片文件 {image_path}")
        print("请确保 'image_path' 的路径正确。")
        exit()
    except Exception as e:
        print(f"发生错误: {e}")
        exit()

    # --- 3. 在图片上绘制红点 ---

    # 设置点的半径和颜色
    dot_radius = 3  # 你可以调整这个值来改变点的大小
    # 注意: OpenCV 使用 BGR 格式，所以红色是 (0, 0, 255)
    dot_color_bgr = (0, 0, 255) 
    dot_thickness = -1 # -1 表示绘制实心圆

    for kp in keypoints:
        x, y = kp[0], kp[1]
        
        # OpenCV 的坐标必须是整数
        center_coordinates = (int(round(x)), int(round(y)))
        
        # 绘制一个实心红点
        cv2.circle(img, center_coordinates, dot_radius, dot_color_bgr, dot_thickness)

    # --- 4. 保存新图片 ---
    cv2.imwrite(output_path, img)

    print(f"绘制完成！新图片已保存到: {output_path}")

