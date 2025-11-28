import os

# --- 配置区域 ---

# 1. 基础检查目录
BASE_DIR = "/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/rgbs"

# 2. 缺失文件的输出文件名
OUTPUT_FILE = "missing_files_rgbs.txt"

# 3. 摄像头索引范围 (Cam{index})
CAM_START_INDEX = 1
CAM_END_INDEX = 160

# 4. 帧索引范围 (xxxx)
FRAME_START_INDEX = 0    # 对应的 "xxxx" 是 "0000"
FRAME_END_INDEX = 2213   # 对应的 "xxxx" 是 "2213"

# --- 脚本正文 ---

print(f"开始检查目录: {BASE_DIR}")
print(f"摄像头范围: {CAM_START_INDEX:03d} 到 {CAM_END_INDEX}")
print(f"帧范围 (xxxx): {FRAME_START_INDEX:04d} 到 {FRAME_END_INDEX:04d}")
print("---")

missing_files_list = []
total_checked_count = 0
total_missing_count = 0

# <--- 新增: 计算总摄像头数，用于显示百分比
total_cameras_to_check = CAM_END_INDEX - CAM_START_INDEX + 1

# 1. 遍历每个摄像头 (例如: 1, 2, ..., 160)
# <--- 修改: 使用 enumerate 来获取当前进度 (i 从 0 开始)
for i, cam_idx in enumerate(range(CAM_START_INDEX, CAM_END_INDEX + 1)):
    
    # 格式化摄像头索引为 "001", "002" ...
    cam_str = f"{cam_idx:03d}"
    cam_dir_name = f"Cam{cam_str}"
    
    # <--- 新增: 重置单个摄像头的缺失计数器
    missing_in_this_cam = 0
    
    # 2. 遍历每一帧 (例如: 0, 1, ..., 2213)
    for frame_idx in range(FRAME_START_INDEX, FRAME_END_INDEX + 1):
        
        # 格式化帧索引为 "0000", "0001", ... "2213"
        frame_str = f"{frame_idx:04d}" # 格式化为4位数 "xxxx"
        
        # 3. 构建文件名和路径
        # 格式: Cam001_rgb000000.jpg, Cam001_rgb000001.jpg ...
        file_name = f"Cam{cam_str}_rgb00{frame_str}.jpg"
        
        # 完整的绝对路径
        full_file_path = os.path.join(BASE_DIR, cam_dir_name, file_name)
        
        # 记录用的相对路径 (更清晰)
        relative_file_path = os.path.join(cam_dir_name, file_name)
        
        # 4. 检查文件是否存在
        total_checked_count += 1
        if not os.path.exists(full_file_path):
            total_missing_count += 1
            missing_in_this_cam += 1 # <--- 修改: 增加单个摄像头的计数
            missing_files_list.append(relative_file_path)

    # <--- 新增: 打印当前摄像头的进度报告 ---
    # (i + 1) 是当前完成的摄像头编号
    progress_percent = (i + 1) / total_cameras_to_check
    
    # :7.2% 是格式化字符串，确保百分比对齐
    print(f"[{cam_dir_name}] 检查完成 (进度: {i + 1}/{total_cameras_to_check} | {progress_percent:7.2%})。 本次发现 {missing_in_this_cam} 个缺失文件。")


# --- 5. 结果汇总与报告 ---

print("---")
print("检查完成。")
print(f"总共应有文件数 (Cameras * Frames): {total_checked_count}")
print(f"检测到缺失文件数: {total_missing_count}")

# 6. 将缺失列表写入TXT文件
try:
    with open(OUTPUT_FILE, "w") as f:
        if not missing_files_list:
            f.write("没有检测到缺失文件。\n")
            print(f"太好了！没有文件缺失。")
        else:
            f.write(f"--- 缺失文件列表 (共 {total_missing_count} 个) ---\n\n")
            for line in missing_files_list:
                f.write(line + "\n")
        print(f"缺失文件列表已保存到: {OUTPUT_FILE}")
except IOError as e:
    print(f"错误：无法写入文件 {OUTPUT_FILE}。原因: {e}")