import numpy as np
import cv2  # OpenCV 用于加载真实的掩码
import open3d as o3d # Open3D 用于可视化和保存点云
import json
import os
from pathlib import Path
import argparse # 用于从命令行读取参数
import matplotlib.pyplot as plt # ❗️ 新增：用于生成颜色

# --- 1. 核心辅助函数 (相机数学) ---
# ... (look_at 和 project_points 函数保持不变) ...
def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    计算一个 "World-to-Camera" (W2C) 4x4 视图矩阵。
    这模拟了将相机放置在 'eye' 点，朝向 'target' 点。
    
    Args:
        eye: (3,) 相机位置
        target: (3,) 相机朝向的目标点
        up: (3,) 相机的“上”方向向量
        
    Returns:
        (4, 4) W2C 矩阵 [R|t]
    """
    # OpenGL 坐标系 (Y 向上, Z 向后)
    z_axis = eye - target # Z 轴朝向观察者
    # 归一化，处理零向量
    norm_z = np.linalg.norm(z_axis)
    if norm_z == 0: norm_z = 1e-6
    z_axis = z_axis / norm_z
    
    x_axis = np.cross(up, z_axis) # X 轴是 "右"
    norm_x = np.linalg.norm(x_axis)
    if norm_x == 0: norm_x = 1e-6
    x_axis = x_axis / norm_x
    
    y_axis = np.cross(z_axis, x_axis) # Y 轴是 "上"

    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    t = -R @ eye
    
    W2C = np.eye(4)
    W2C[:3, :3] = R
    W2C[:3, 3] = t
    return W2C

def project_points(points_3d: np.ndarray, K: np.ndarray, W2C: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    辅助函数：将 3D 世界点批量投影到 2D 像素坐标。
    (注意：此函数不处理镜头畸变)

    Args:
        points_3d (np.ndarray): (M, 3) 3D 世界点
        K (np.ndarray): (3, 3) 相机内参
        W2C (np.ndarray): (4, 4) 世界到相机外参

    Returns:
        pixels_2d (np.ndarray): (M, 2) 像素坐标 (u, v)
        cam_z (np.ndarray): (M,) 深度 (Z值)
    """
    
    # 将 (M, 3) 转换为 (M, 4) 齐次坐标
    points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # 构造投影矩阵 P = K @ [R|t]
    # [R|t] 是 W2C 的前 3x4 部分
    P = K @ W2C[:3, :] # (3, 3) @ (3, 4) -> (3, 4)
    
    # 批量投影: (3, 4) @ (M, 4).T -> (3, M) -> (M, 3)
    pixels_homo = (P @ points_homo.T).T 
    
    # 归一化 (除以深度)
    w = pixels_homo[:, 2] 
    w[w == 0] = 1e-6 # 防止除以零
    
    pixels_2d = pixels_homo[:, :2] / w[:, None] # (M, 2)
    
    return pixels_2d, w


# --- 2. 空间雕刻主算法 (交集) ---
def space_carving(cameras: dict, bbox: np.ndarray, resolution: int = 128) -> np.ndarray:
    """
    执行空间雕刻 (Space Carving) 算法 (交集)。
    """
    
    # 1. 初始化体素网格
    N = resolution
    x_range = np.linspace(bbox[0, 0], bbox[1, 0], N)
    y_range = np.linspace(bbox[0, 1], bbox[1, 1], N)
    z_range = np.linspace(bbox[0, 2], bbox[1, 2], N)
    
    # 使用 'ij' 索引以匹配 (H, W, D) -> (y, x, z)
    grid_y, grid_x, grid_z = np.meshgrid(y_range, x_range, z_range, indexing='ij')
    
    # 将它们堆叠为 (H, W, D, 3) 的网格
    voxel_centers_grid = np.stack([grid_x, grid_y, grid_z], axis=-1)
    
    # 2. 初始化占用图 (Occupancy Grid)
    occupancy_grid = np.ones((N, N, N), dtype=bool)
    voxel_centers_flat = voxel_centers_grid.reshape(-1, 3) # (N*N*N, 3)
    
    print(f"开始雕刻 {voxel_centers_flat.shape[0]} 个体素...")

    # 3. 循环 "雕刻"
    total_cameras = len(cameras)
    if total_cameras == 0:
        print("警告：没有相机可用于雕刻。请检查你的 --cams 或 --skip_cams 参数。")
        return np.array([]) # 返回空点云

    for i, (cam_id, cam_data) in enumerate(cameras.items()):
        K = cam_data['K']
        W2C = cam_data['W2C']
        mask = cam_data['mask']
        H, W = mask.shape
        
        # a. 批量投影所有体素
        pixels_2d, cam_z = project_points(voxel_centers_flat, K, W2C)
        
        # b. 检查一致性
        u = pixels_2d[:, 0]
        v = pixels_2d[:, 1]
        
        # c. 深度检查 (OpenGL: 必须在相机前面, Z 为负)
        in_front = (cam_z < 0)
        
        # d. 边界检查 (必须在图像范围内)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        
        # e. 将有效投影的索引计算回整数坐标
        valid_indices = in_front & in_bounds
        
        # f. 掩码检查 (只对在边界内的点进行)
        is_consistent = np.zeros(voxel_centers_flat.shape[0], dtype=bool) # 默认所有点都无效
        
        u_int = np.round(u[valid_indices]).astype(int)
        v_int = np.round(v[valid_indices]).astype(int)
        
        # 将索引裁剪到 [0, W-1] 和 [0, H-1] 的有效范围内
        u_int = np.clip(u_int, 0, W - 1)
        v_int = np.clip(v_int, 0, H - 1)

        # (v, u) 对应 (row, col)
        is_consistent[valid_indices] = mask[v_int, u_int]
        
        # g. 更新占用图
        is_consistent_grid = is_consistent.reshape(N, N, N)
        
        # (N,N,N) & (N,N,N) -> 雕刻
        occupancy_grid = occupancy_grid & is_consistent_grid 
            
        print(f"  [{i+1}/{total_cameras}] 处理相机 {cam_id}, 幸存: {np.sum(occupancy_grid)} 体素")
        
        if np.sum(occupancy_grid) == 0:
            print(f"  [!] 幸存体素为 0，在相机 {cam_id} 处提前停止雕刻。")
            break

    # 4. 提取点云
    final_point_cloud = voxel_centers_grid[occupancy_grid]
    
    print(f"雕刻完成！最终视觉外壳包含 {final_point_cloud.shape[0]} 个点。")
    return final_point_cloud


# --- 3. 新增函数：可视化单独的视觉外壳 (并集) ---
def visualize_individual_hulls(cameras: dict, bbox: np.ndarray, resolution: int = 128, sample_perc: float = 0.1) -> (np.ndarray, np.ndarray):
    """
    计算每个相机的视觉外壳，为它们着色，并返回一个合并的点云。
    
    Args:
        sample_perc (float): 对未雕刻的灰色体素的采样百分比 (0.0 到 1.0)
    
    Returns:
        points (np.ndarray): (TotalPoints, 3)
        colors (np.ndarray): (TotalPoints, 3) in [0, 1] RGB
    """
    
    # 1. 初始化体素网格
    N = resolution
    x_range = np.linspace(bbox[0, 0], bbox[1, 0], N)
    y_range = np.linspace(bbox[0, 1], bbox[1, 1], N)
    z_range = np.linspace(bbox[0, 2], bbox[1, 2], N)
    grid_y, grid_x, grid_z = np.meshgrid(y_range, x_range, z_range, indexing='ij')
    voxel_centers_grid = np.stack([grid_x, grid_y, grid_z], axis=-1)
    voxel_centers_flat = voxel_centers_grid.reshape(-1, 3) # (N*N*N, 3)
    
    print(f"开始为 {voxel_centers_flat.shape[0]} 个体素计算单独的视觉外壳...")

    # 2. 准备颜色
    total_cameras = len(cameras)
    if total_cameras == 0:
        print("警告：没有相机可用于可视化。")
        return np.array([]), np.array([])
        
    # 使用 'hsv' 颜色映射表为每个相机生成唯一的颜色
    cmap = plt.get_cmap('hsv')
    
    final_points_list = []
    final_colors_list = []

    # ❗️ 新增：一个布尔掩码，用于跟踪 *所有* 被雕刻的体素
    all_carved_voxels_mask = np.zeros(voxel_centers_flat.shape[0], dtype=bool)

    # 3. 循环处理每个相机
    for i, (cam_id, cam_data) in enumerate(cameras.items()):
        K = cam_data['K']
        W2C = cam_data['W2C']
        mask = cam_data['mask']
        H, W = mask.shape
        
        # a. 投影
        pixels_2d, cam_z = project_points(voxel_centers_flat, K, W2C)
        
        # b. 检查
        u = pixels_2d[:, 0]
        v = pixels_2d[:, 1]
        in_front = (cam_z < 0)
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid_indices = in_front & in_bounds
        
        # c. 掩码检查
        is_consistent = np.zeros(voxel_centers_flat.shape[0], dtype=bool)
        u_int = np.round(u[valid_indices]).astype(int)
        v_int = np.round(v[valid_indices]).astype(int)
        u_int = np.clip(u_int, 0, W - 1)
        v_int = np.clip(v_int, 0, H - 1)
        
        # 找到只属于这个相机的点
        is_consistent[valid_indices] = mask[v_int, u_int]
        
        # ❗️ 新增：更新“所有被雕刻体素”的并集
        all_carved_voxels_mask = all_carved_voxels_mask | is_consistent
        
        # d. 提取点云
        cam_points = voxel_centers_flat[is_consistent]
        
        if cam_points.shape[0] == 0:
            print(f"  [{i+1}/{total_cameras}] 相机 {cam_id} 未雕刻出任何点 (可能是 BBox 错位)。")
            continue
            
        # e. 为这些点生成颜色
        color_rgba = cmap(i / total_cameras) # (R, G, B, A) in [0, 1]
        color_rgb = np.array(color_rgba[:3]) # (3,)
        cam_colors = np.tile(color_rgb, (cam_points.shape[0], 1))
        
        # f. 添加到列表
        final_points_list.append(cam_points)
        final_colors_list.append(cam_colors)
        
        print(f"  [{i+1}/{total_cameras}] 处理相机 {cam_id}, 找到 {cam_points.shape[0]} 个体素。")

    # ❗️ 修正：计算并 *采样* 灰色点
    gray_points_mask = ~all_carved_voxels_mask
    gray_indices = np.where(gray_points_mask)[0] # 获取所有灰色点的索引
    
    if gray_indices.shape[0] > 0:
        # 计算要采样的数量
        num_to_sample = int(gray_indices.shape[0] * sample_perc)
        
        # 仅在 num_to_sample 大于 0 时才采样
        if num_to_sample > 0:
            print(f"  [*] {gray_indices.shape[0]} 个灰色体素中，随机采样 {sample_perc*100:.1f}% ({num_to_sample} 个点)...")
            
            # 随机选择索引
            sampled_indices = np.random.choice(gray_indices, size=num_to_sample, replace=False)
            
            # 获取采样后的点
            gray_points = voxel_centers_flat[sampled_indices]
            
            if gray_points.shape[0] > 0:
                gray_colors = np.full((gray_points.shape[0], 3), 0.5) # 灰色 [0.5, 0.5, 0.5]
                final_points_list.append(gray_points)
                final_colors_list.append(gray_colors)
        else:
            print(f"  [*] {gray_indices.shape[0]} 个灰色体素。采样率为 {sample_perc*100:.1f}%, 采样 0 个点。")


    if not final_points_list:
        print("可视化完成！未找到任何点。")
        return np.array([]), np.array([])

    # 4. 合并所有点云
    all_points = np.vstack(final_points_list)
    all_colors = np.vstack(final_colors_list)
    
    print(f"可视化完成！总共合并了 {all_points.shape[0]} 个点。")
    return all_points, all_colors


# --- 4. 加载 transforms.json ---
# ❗️ 修正：添加 flip_x_axis 和 use_undistortion
def load_data_from_json(json_path: Path, base_path: Path, skip_cams: list = None, flip_x_axis: bool = False, use_undistortion: bool = True) -> dict:
    """
    从 transforms.json 加载相机参数和掩码。
    
    Args:
        skip_cams (list): 要跳过的相机标签列表 (例如 ['11', '12'])
        flip_x_axis (bool): 是否翻转 X 轴以匹配左手坐标系的 GT
        use_undistortion (bool): 是否应用镜头畸变矫正
    """
    print(f"正在从 {json_path} 加载数据...")
    if skip_cams:
        print(f"将跳过相机: {skip_cams}")
        
    cameras = {}
    
    # ❗️ 新增：用于匹配 GT 的 X 轴翻转矩阵
    gt_flip = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'frames' not in data:
        print(f"错误: JSON 文件 {json_path} 中未找到 'frames' 键。")
        return {}

    for frame in data['frames']:
        cam_label = frame['camera_label'] # e.g., "00"

        if skip_cams and cam_label in skip_cams:
            print(f"  [*] 已跳过相机 {cam_label}")
            continue

        # 1. 加载内参
        K = np.array([
            [frame['fl_x'], 0, frame['cx']],
            [0, frame['fl_y'], frame['cy']],
            [0, 0, 1]
        ], dtype=np.float32)

        # 2. 加载外参
        C2W = np.array(frame['transform_matrix'], dtype=np.float32)
        
        # ❗️ 新增：如果 flip_x_axis 为 True，则应用翻转
        if flip_x_axis:
            # 这会将相机位姿转换到一个 X 轴翻转的世界
            C2W = gt_flip @ C2W
        
        W2C = np.linalg.inv(C2W)
        
        # 3. 加载掩码
        file_path = Path(frame['file_path']) 
        
        if len(file_path.parts) < 2:
            print(f"警告: file_path '{file_path}' 格式不正确。跳过相机 {cam_label}。")
            continue
            
        cam_dir_name = file_path.parts[-2] 
        mask_filename = file_path.parts[-1] 
        mask_filename = Path(mask_filename).with_suffix(".png")
        mask_path = base_path / "fmasks" / cam_dir_name / mask_filename
        
        if not mask_path.exists():
            print(f"警告: 找不到推断出的掩码 {mask_path}。跳过相机 {cam_label}。")
            continue
            
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"警告: 无法读取掩码 {mask_path}。跳过相机 {cam_label}。")
            continue
            
        if np.all(mask_img <= 128): # (使用 <= 128 以防万一)
            print(f"警告: 掩码 {mask_path} 似乎是空的(全黑)。跳过相机 {cam_label}。")
            continue
            
        # ❗️ --- 关键修正：处理镜头畸变 --- ❗️
        # 1. 从 JSON 加载畸变系数
        k1 = frame.get('k1', 0.0)
        k2 = frame.get('k2', 0.0)
        p1 = frame.get('p1', 0.0)
        p2 = frame.get('p2', 0.0)
        k3 = frame.get('k3', 0.0) 
        distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        # ❗️ 修正：现在由 'use_undistortion' 开关控制
        if use_undistortion and np.any(distCoeffs != 0):
            print(f"  [*] 正在为相机 {cam_label} 矫正畸变掩码...")
            # 注意：cv2.undistort 需要 uint8 图像
            mask_img_undistorted = cv2.undistort(mask_img.astype(np.uint8), K, distCoeffs)
            mask_bool = mask_img_undistorted > 128
        else:
            # 如果没有畸变，或用户禁用了矫正，则正常转换
            if not use_undistortion:
                print(f"  [*] 已跳过相机 {cam_label} 的畸变矫正 (用户禁用)。")
            mask_bool = mask_img > 128
        # ❗️ --- 畸变处理结束 --- ❗️

        # 4. 存储所有数据
        orint_cam_label = frame.get('orint_camera_label', cam_label)
        
        cameras[cam_label] = {
            'K': K, 
            'W2C': W2C,
            'C2W': C2W, # ❗️ 存储 C2W 用于可视化 (现在可能是翻转过的)
            'mask': mask_bool, 
            'label': cam_label,
            'orint_label': orint_cam_label
        }

    print(f"成功加载了 {len(cameras)} 个相机的数据。")
    return cameras


# ❗️ --- 新增：用于可视化相机的函数 --- ❗️
def create_camera_visualization(cameras: dict, ray_length: float = 1.0, num_ray_points: int = 10) -> (np.ndarray, np.ndarray):
    """
    为所有相机创建点云几何体（圆点 + 射线）。
    
    Args:
        cameras (dict): 加载的相机字典。
        ray_length (float): 可视化射线的长度（米）。
        num_ray_points (int): 每条射线上采样的点数。
        
    Returns:
        points (np.ndarray): (NumCams * (1 + NumPoints), 3)
        colors (np.ndarray): (NumCams * (1 + NumPoints), 3) [0, 1] RGB
    """
    
    all_cam_points = []
    all_cam_colors = []
    
    # 为相机射线设置一个独特的颜色 (例如，白色)
    cam_color = np.array([1.0, 1.0, 1.0]) # [R, G, B]
    
    for cam_id, cam_data in cameras.items():
        C2W = cam_data['C2W'] # (4, 4)
        
        # 1. "圆点" (相机位置)
        # C2W 的平移向量
        cam_pos = C2W[:3, 3]
        all_cam_points.append(cam_pos)
        all_cam_colors.append(cam_color)

        # 2. "射线" (相机朝向)
        # OpenGL: 相机朝向 -Z 轴
        # C2W 的 Z 轴是第三列
        view_dir = -C2W[:3, 2]
        
        # 归一化（以防万一）
        norm_view_dir = np.linalg.norm(view_dir)
        if norm_view_dir == 0: norm_view_dir = 1e-6
        view_dir = view_dir / norm_view_dir
        
        # 沿着射线采样点
        ray_steps = np.linspace(0.0, ray_length, num_ray_points)
        ray_points = cam_pos[None, :] + ray_steps[:, None] * view_dir[None, :]
        
        all_cam_points.append(ray_points)
        all_cam_colors.append(np.tile(cam_color, (num_ray_points, 1)))

    if not all_cam_points:
        return np.array([]), np.array([])
        
    return np.vstack(all_cam_points), np.vstack(all_cam_colors)


# --- 5. 主程序入口 ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="使用多视图掩码进行空间雕刻以提取视觉外壳。")
    
    # ❗️ 路径已更新
    parser.add_argument("--json", type=Path, default=Path('/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio/transforms_input.json'),
                        help="指向 transforms.json 文件的路径。")
    parser.add_argument("--path", type=Path, default=Path('/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio'),
                        help="数据集的根目录 (包含 fmasks/ 文件夹)。")
    
    # ❗️ BBox 默认值已更新 (以 Y=1.0 为中心, 高度 2.0m, 宽度/深度 3.0m)
    parser.add_argument("--bbox_min", type=float, nargs=3, default=[-3, -3, -3],
                        help="雕刻空间的 BBox 最小值 (x y z)。例如: -1.5 0.0 -1.5")
    parser.add_argument("--bbox_max", type=float, nargs=3, default=[3, 3, 3],
                        help="雕刻空间的 BBox 最大值 (x y z)。例如: 1.5 2.0 1.5")
    
    # ❗️ 分辨率默认值已更新
    parser.add_argument("--res", type=int, default=256, 
                        help="体素网格的分辨率 (例如: 128 将创建 128x128x128 网格)。")
    
    # ❗️ .ply 输出路径已更新
    parser.add_argument("--output_ply", type=Path, default=Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio/sparse_pcd.ply"),
                        help="保存最终 .ply 点云的文件路径。")
    
    parser.add_argument("--skip_cams", type=str, nargs='+', default=None,
                        help="要跳过的相机标签列表 (例如: 11 12 13)")

    parser.add_argument("--max_cams", type=int, default=None,
                        help="只处理前 N 个相机 (用于调试)。默认：None (处理所有相机)。")
    
    # ❗️ ["01","13"]     "22", "11"
    parser.add_argument("--cams", type=str, nargs='+', default= ["36", "29","22", "11"],
                        help="要用于雕刻的特定相机标签列表 (例如: 00 01 05)")
    
    parser.add_argument("--mode", type=str, default="carve", choices=['carve', 'visualize'],
                        help="要执行的操作：'visualize' (交集) 或 'visualize' (彩色并集)。默认: visualize")
    
    # ❗️ --- 修正：--sample_gray_perc 默认值改为 0.05 (5%) --- ❗️
    parser.add_argument("--sample_gray_perc", type=float, default=0,
                        help="对未雕刻的灰色体素进行随机采样的百分比 (0.0 到 1.0)。默认: 0.05 (5%%)")

    # ❗️ --- 新增：--vis_cams --- ❗️
    parser.add_argument("--vis_cams", action="store_true",
                        help="在 .ply 文件中添加相机位置 (圆点) 和朝向 (射线)。")

    # ❗️ --- 新增：--cam_ray_length --- ❗️
    parser.add_argument("--cam_ray_length", type=float, default=1.0,
                        help="可视化相机射线的长度（米）。默认: 1.0")

    # ❗️ --- 新增：--flip_x_axis --- ❗️
    parser.add_argument("--flip_x_axis", default=True, type=bool,
                        help="翻转世界 X 轴以匹配左手坐标系 (例如，匹配 GT)。")

    # ❗️ --- 关键修正：将两个畸变参数合并为一个 --- ❗️
    parser.add_argument("--undistort", type=bool, default=False,
                        help="禁用镜头畸变矫正 (默认: 启用)")


    args = parser.parse_args()

    # 1. 从 JSON 加载真实相机数据和掩码
    # ❗️ 修正：传入 flip_x_axis 和 use_undistortion 标志
    cameras = load_data_from_json(args.json, args.path, 
                                  skip_cams=args.skip_cams, 
                                  flip_x_axis=args.flip_x_axis,
                                  use_undistortion=args.undistort)
    
    if not cameras:
        print("未能加载任何相机数据。正在退出。")
        exit()
        
    # ❗️ --- 根据 --cams 过滤相机 --- ❗️
    if args.cams:
        print(f"--- 警告: 将只使用 --cams 指定的 {len(args.cams)} 个相机 ---")
        filtered_cameras = {}
        for cam_key in args.cams:
            if cam_key in cameras:
                filtered_cameras[cam_key] = cameras[cam_key]
            else:
                print(f"  [!] 警告: --cams 中指定的相机 '{cam_key}' 在 JSON 中未找到或加载失败。")
        cameras = filtered_cameras
    
    # ❗️ --- 根据 max_cams 裁剪相机列表 --- ❗️
    if args.max_cams is not None and args.max_cams > 0:
        print(f"--- 警告: 将只处理前 {args.max_cams} 个相机 ---")
        limited_cam_keys = list(cameras.keys())[:args.max_cams]
        cameras = {key: cameras[key] for key in limited_cam_keys}
    # ❗️ --- 裁剪结束 --- ❗️

    # 2. 准备 BBox
    bbox = np.array([
        args.bbox_min,
        args.bbox_max
    ])
    
    # 3. 检查模式 (已移除 --debug_cam)
    if args.mode == 'carve':
        # 3a. (正常) 执行空间雕刻算法
        point_cloud = space_carving(cameras, bbox, resolution=args.res)
        points = point_cloud
        colors = None # 雕刻模式不生成颜色
        
    else: # 默认为 'visualize'
        # 3b. (调试) 执行可视化算法
        points, colors = visualize_individual_hulls(cameras, bbox, 
                                                    resolution=args.res, 
                                                    sample_perc=args.sample_gray_perc)
        
    # ❗️ --- 新增：添加相机可视化 --- ❗️
    if args.vis_cams:
        print(f"正在为 {len(cameras)} 个相机创建可视化几何体...")
        # ❗️ 修正：使用 args.cam_ray_length
        cam_points, cam_colors = create_camera_visualization(cameras, ray_length=args.cam_ray_length)
        
        if points.shape[0] > 0:
            points = np.vstack([points, cam_points])
            if colors is not None:
                colors = np.vstack([colors, cam_colors])
            else:
                # 'carve' 模式没有颜色，所以我们只给相机上色
                # (这有点难办，但我们可以给雕刻的点一个默认色)
                carve_colors = np.full((points.shape[0], 3), 0.8) # 浅灰色
                colors = np.vstack([carve_colors, cam_colors])
        else:
            # 只有相机
            points = cam_points
            colors = cam_colors
    # ❗️ --- 可视化结束 --- ❗️

    # --- 4. 保存点云为 PLY 文件 ---
    if points.shape[0] > 0:
        # 确保父目录存在
        output_file = args.output_ply.resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"正在将 {points.shape[0]} 个点保存到 {output_file}...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # ❗️ 修正：只有在 'visualize' 模式或 'vis_cams' 模式下才添加颜色
        if colors is not None and colors.shape[0] == points.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(str(output_file), pcd)
        print(f"成功保存到 {output_file}")
    else:
        print("点云为空，未保存 .ply 文件。")