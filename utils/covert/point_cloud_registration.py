#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云配准工具
用于将点云C（来自相机阵列A）对齐到点云D（来自相机阵列B）的位置
支持两种方法：
1. 使用ICP算法进行自动配准（无需相机外参）
2. 使用相机外参计算变换矩阵（如果有相机标定数据）
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple, List
import cv2 as cv
import json
from scipy.spatial.transform import Rotation

from utils.covert.camera_data import CameraData, read_calibration_csv


def read_transforms_json(json_path: Path) -> List[CameraData]:
    """
    从nerfstudio格式的transforms.json文件读取相机参数
    
    Args:
        json_path: transforms.json文件路径
        
    Returns:
        List[CameraData]: 相机数据列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cameras = []
    frames = data.get('frames', [])
    
    for idx, frame in enumerate(frames):
        # 提取transform_matrix (camera-to-world)
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        
        # 提取内参
        fl_x = float(frame['fl_x'])
        fl_y = float(frame['fl_y'])
        cx = float(frame['cx'])
        cy = float(frame['cy'])
        w = int(frame['w'])
        h = int(frame['h'])
        
        # 提取畸变系数（如果有）
        k1 = float(frame.get('k1', 0.0))
        k2 = float(frame.get('k2', 0.0))
        k3 = float(frame.get('k3', 0.0))
        
        # 从c2w矩阵提取旋转和平移
        # c2w是4x4矩阵，前3x3是旋转，前3x1的最后一列是平移
        rotation_matrix = c2w[:3, :3]
        translation = c2w[:3, 3]
        
        # 将旋转矩阵转换为轴角表示
        rotation = Rotation.from_matrix(rotation_matrix)
        rotation_axisangle = rotation.as_rotvec()
        
        # 计算归一化的焦距和主点
        focal_length = np.array([fl_x / w, fl_y / h])
        principal_point = np.array([cx / w, cy / h])
        
        # 生成相机名称（优先使用camera_label，其次colmap_im_id，最后索引）
        camera_name = frame.get('camera_label')
        if camera_name is None:
            camera_name = frame.get('colmap_im_id')
        if camera_name is None:
            camera_name = f"Cam{idx:03d}"
        camera_name = str(camera_name)
        
        camera = CameraData(
            name=str(camera_name),
            width=w,
            height=h,
            rotation_axisangle=rotation_axisangle,
            translation=translation,
            focal_length=focal_length,
            principal_point=principal_point,
            k1=k1,
            k2=k2,
            k3=k3
        )
        cameras.append(camera)
    
    return cameras


def load_point_cloud(ply_path: Path) -> o3d.geometry.PointCloud:
    """
    加载点云文件（PLY格式）
    
    Args:
        ply_path: PLY文件路径
        
    Returns:
        open3d.geometry.PointCloud: 点云对象
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError(f"无法加载点云文件: {ply_path}")
    return pcd


def register_point_clouds_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
    max_iterations: int = 30,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    使用ICP算法将源点云配准到目标点云
    
    Args:
        source_pcd: 源点云（点云C）
        target_pcd: 目标点云（点云D）
        voxel_size: 下采样体素大小（米）
        max_iterations: 最大迭代次数
        verbose: 是否打印详细信息
        
    Returns:
        transform_matrix (4x4): 从源点云到目标点云的变换矩阵
        fitness: 配准质量分数 (0-1)
    """
    if verbose:
        print(f"源点云点数: {len(source_pcd.points)}")
        print(f"目标点云点数: {len(target_pcd.points)}")
    
    # 1. 下采样以提高配准速度
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    if verbose:
        print(f"下采样后源点云点数: {len(source_down.points)}")
        print(f"下采样后目标点云点数: {len(target_down.points)}")
    
    # 2. 估计法向量（ICP需要法向量）
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # 3. 粗配准（使用FPFH特征）
    if verbose:
        print("正在进行粗配准...")
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    
    # RANSAC粗配准
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    if verbose:
        print(f"粗配准完成，fitness: {result_ransac.fitness:.4f}")
    
    # 4. 精配准（ICP）
    if verbose:
        print("正在进行精配准（ICP）...")
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance=voxel_size * 1.5,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iterations
        )
    )
    
    if verbose:
        print(f"精配准完成，fitness: {result_icp.fitness:.4f}, inlier_rmse: {result_icp.inlier_rmse:.6f}")
    
    return result_icp.transformation, result_icp.fitness


def compute_transform_from_array_geometry(
    source_cameras: List[CameraData],
    target_cameras: List[CameraData]
) -> np.ndarray:
    """
    通过分析两个相机阵列的几何结构计算变换矩阵（无需公共相机）
    
    该方法适用于两个相机阵列没有重叠相机的情况。
    通过分析相机阵列的几何结构（质心、主方向等）来估计初始变换。
    
    Args:
        source_cameras: 源相机阵列的相机数据列表
        target_cameras: 目标相机阵列的相机数据列表
        
    Returns:
        transform_matrix (4x4): 从源点云坐标系到目标点云坐标系的初始变换矩阵
    """
    
    if len(source_cameras) < 3 or len(target_cameras) < 3:
        raise ValueError("每个相机阵列至少需要3个相机才能使用几何结构方法")
    
    # 提取相机中心位置
    source_centers = np.array([cam.translation for cam in source_cameras])
    target_centers = np.array([cam.translation for cam in target_cameras])
    
    # 计算质心
    source_centroid = np.mean(source_centers, axis=0)
    target_centroid = np.mean(target_centers, axis=0)
    
    # 计算主方向（使用PCA）
    source_centered = source_centers - source_centroid
    target_centered = target_centers - target_centroid
    
    # 使用SVD计算主方向
    source_U, source_S, source_Vt = np.linalg.svd(source_centered.T, full_matrices=False)
    target_U, target_S, target_Vt = np.linalg.svd(target_centered.T, full_matrices=False)
    
    # 主方向（前3个主成分）
    source_principal = source_Vt[:3, :].T  # 每行是一个主方向向量
    target_principal = target_Vt[:3, :].T
    
    # 使用主方向对齐来计算旋转
    # 方法：将源阵列的主方向对齐到目标阵列的主方向
    # 注意：需要处理方向向量的符号（PCA方向可能相反）
    
    # 构建方向矩阵（每列是一个主方向）
    source_dirs = source_principal.T  # 3x3，每列是一个主方向
    target_dirs = target_principal.T
    
    # 使用Kabsch算法计算旋转
    H = source_dirs @ target_dirs.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 确保是右手坐标系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算平移
    t = target_centroid - R @ source_centroid
    
    # 构建4x4变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t
    
    print(f"使用几何结构方法计算初始变换（源阵列{len(source_cameras)}个相机，目标阵列{len(target_cameras)}个相机）")
    return transform_matrix


def compute_transform_from_cameras(
    source_calibration_path: Path,
    target_calibration_path: Path,
    reference_camera_name: str = "Cam127",
    use_multiple_cameras: bool = False,
    common_camera_names: Optional[List[str]] = None,
    use_array_geometry: bool = False,
    source_is_json: bool = False,
    target_is_json: bool = False
) -> np.ndarray:
    """
    通过相机外参计算从源点云到目标点云的变换矩阵
    
    该方法假设：
    - 源点云C在相机阵列A的世界坐标系中
    - 目标点云D在相机阵列B的世界坐标系中
    - 两个相机阵列拍摄的是同一个物体，但处于不同的世界坐标系
    
    方法说明：
    1. 单相机方法（默认）：使用一个公共参考相机计算变换
    2. 多相机方法：使用多个公共相机计算更准确的变换（通过最小化重投影误差）
    3. 几何结构方法：当没有公共相机时，使用相机阵列的几何结构计算变换
    
    Args:
        source_calibration_path: 源相机阵列（A）的标定文件（CSV或JSON）
        target_calibration_path: 目标相机阵列（B）的标定文件（CSV或JSON）
        reference_camera_name: 用于对齐的参考相机名称（单相机方法）
        use_multiple_cameras: 是否使用多相机方法
        common_camera_names: 公共相机名称列表（多相机方法，如果为None则自动查找）
        use_array_geometry: 是否使用几何结构方法（当没有公共相机时）
        source_is_json: 源文件是否为JSON格式
        target_is_json: 目标文件是否为JSON格式
        
    Returns:
        transform_matrix (4x4): 从源点云坐标系到目标点云坐标系的变换矩阵
    """
    # 读取两个相机阵列的标定数据
    if source_is_json:
        source_cameras = read_transforms_json(source_calibration_path)
    else:
        source_cameras = read_calibration_csv(source_calibration_path)
    
    if target_is_json:
        target_cameras = read_transforms_json(target_calibration_path)
    else:
        target_cameras = read_calibration_csv(target_calibration_path)
    
    # 创建相机名称到相机对象的映射
    source_cam_dict = {cam.name: cam for cam in source_cameras}
    target_cam_dict = {cam.name: cam for cam in target_cameras}
    
    # 检查是否有公共相机
    common_cameras_found = [name for name in source_cam_dict.keys() 
                           if name in target_cam_dict.keys()]
    
    # 如果没有公共相机，自动使用几何结构方法
    if len(common_cameras_found) == 0:
        if use_array_geometry or (not use_multiple_cameras and reference_camera_name not in source_cam_dict):
            print("未找到公共相机，自动使用几何结构方法计算变换...")
            return compute_transform_from_array_geometry(source_cameras, target_cameras)
        else:
            raise ValueError(f"两个相机阵列没有公共相机。"
                           f"源相机: {list(source_cam_dict.keys())[:5]}..."
                           f"目标相机: {list(target_cam_dict.keys())[:5]}..."
                           f"\n请使用 --use_array_geometry 选项或确保有公共相机。")
    
    if use_array_geometry:
        print("使用几何结构方法计算变换...")
        return compute_transform_from_array_geometry(source_cameras, target_cameras)
    
    if use_multiple_cameras:
        # 多相机方法：使用多个公共相机计算更准确的变换
        if common_camera_names is None:
            # 自动查找公共相机
            common_camera_names = [name for name in source_cam_dict.keys() 
                                  if name in target_cam_dict.keys()]
            if len(common_camera_names) == 0:
                print(f"警告：未找到公共相机，改用几何结构方法")
                return compute_transform_from_array_geometry(source_cameras, target_cameras)
            print(f"找到 {len(common_camera_names)} 个公共相机: {common_camera_names[:5]}...")
        
        # 使用多个相机的位置计算变换
        # 方法：通过最小化相机中心点之间的误差来计算变换
        source_centers = []
        target_centers = []
        
        for cam_name in common_camera_names:
            if cam_name in source_cam_dict and cam_name in target_cam_dict:
                source_cam = source_cam_dict[cam_name]
                target_cam = target_cam_dict[cam_name]
                
                # 相机中心在世界坐标系中的位置（translation就是相机中心）
                source_centers.append(source_cam.translation)
                target_centers.append(target_cam.translation)
        
        if len(source_centers) < 3:
            print(f"警告：公共相机数量少于3个，改用单相机方法")
            use_multiple_cameras = False
        else:
            source_centers = np.array(source_centers)
            target_centers = np.array(target_centers)
            
            # 计算质心
            source_centroid = np.mean(source_centers, axis=0)
            target_centroid = np.mean(target_centers, axis=0)
            
            # 去中心化
            source_centered = source_centers - source_centroid
            target_centered = target_centers - target_centroid
            
            # 使用SVD计算旋转矩阵（Kabsch算法）
            H = source_centered.T @ target_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # 确保是右手坐标系（det(R) = 1）
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # 计算平移
            t = target_centroid - R @ source_centroid
            
            # 构建4x4变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            print(f"使用 {len(source_centers)} 个公共相机计算变换矩阵")
            return transform_matrix
    
    # 单相机方法：使用参考相机计算变换
    if reference_camera_name not in source_cam_dict:
        available_cameras = list(source_cam_dict.keys())[:5]
        raise ValueError(f"在源相机标定文件中找不到相机: {reference_camera_name}。"
                        f"可用相机: {available_cameras}")
    if reference_camera_name not in target_cam_dict:
        available_cameras = list(target_cam_dict.keys())[:5]
        raise ValueError(f"在目标相机标定文件中找不到相机: {reference_camera_name}。"
                        f"可用相机: {available_cameras}")
    
    source_ref_cam = source_cam_dict[reference_camera_name]
    target_ref_cam = target_cam_dict[reference_camera_name]
    
    # 计算相机到世界的变换矩阵（C2W）
    source_c2w = source_ref_cam.extrinsic_matrix_cam2world()
    target_c2w = target_ref_cam.extrinsic_matrix_cam2world()
    
    # 计算从源世界坐标系到目标世界坐标系的变换
    # 假设两个相机阵列的参考相机在物理上处于相同位置（或已知相对位置）
    # 如果两个参考相机都指向同一个物体，我们可以通过它们的相对变换计算坐标系变换
    # 
    # 方法：计算两个参考相机之间的相对变换
    # source_world = source_c2w @ camera_space
    # target_world = target_c2w @ camera_space
    # 因此：target_world = target_c2w @ inv(source_c2w) @ source_world
    
    transform_matrix = target_c2w @ np.linalg.inv(source_c2w)
    
    print(f"使用参考相机 {reference_camera_name} 计算变换矩阵")
    return transform_matrix


def apply_transform_to_point_cloud(
    pcd: o3d.geometry.PointCloud,
    transform_matrix: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    将变换矩阵应用到点云
    
    Args:
        pcd: 输入点云
        transform_matrix: 4x4变换矩阵
        
    Returns:
        变换后的点云
    """
    transformed_pcd = pcd.copy()
    transformed_pcd.transform(transform_matrix)
    return transformed_pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: Path):
    """
    保存点云到PLY文件
    
    Args:
        pcd: 点云对象
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"点云已保存到: {output_path}")


def register_and_align_point_clouds(
    source_pcd_path: Path,
    target_pcd_path: Path,
    output_pcd_path: Path,
    method: str = "icp",
    source_calibration_path: Optional[Path] = None,
    target_calibration_path: Optional[Path] = None,
    reference_camera_name: str = "Cam127",
    use_multiple_cameras: bool = False,
    common_camera_names: Optional[List[str]] = None,
    use_array_geometry: bool = False,
    source_is_json: bool = False,
    target_is_json: bool = False,
    voxel_size: float = 0.05,
    max_iterations: int = 30
) -> Tuple[np.ndarray, Optional[float]]:
    """
    将源点云配准到目标点云并保存结果
    
    Args:
        source_pcd_path: 源点云文件路径（点云C）
        target_pcd_path: 目标点云文件路径（点云D）
        output_pcd_path: 输出对齐后的点云文件路径
        method: 配准方法，"icp" 或 "camera"
        source_calibration_path: 源相机阵列的标定文件（method="camera"时需要，CSV或JSON）
        target_calibration_path: 目标相机阵列的标定文件（method="camera"时需要，CSV或JSON）
        reference_camera_name: 参考相机名称（method="camera"单相机方法时使用）
        use_multiple_cameras: 是否使用多相机方法（method="camera"时）
        common_camera_names: 公共相机名称列表（多相机方法，None则自动查找）
        use_array_geometry: 是否使用几何结构方法（当没有公共相机时）
        source_is_json: 源标定文件是否为JSON格式
        target_is_json: 目标标定文件是否为JSON格式
        voxel_size: ICP下采样体素大小
        max_iterations: ICP最大迭代次数
        
    Returns:
        transform_matrix: 变换矩阵
        fitness: 配准质量分数（仅ICP方法，camera方法返回None）
    """
    print(f"加载源点云: {source_pcd_path}")
    source_pcd = load_point_cloud(source_pcd_path)
    
    print(f"加载目标点云: {target_pcd_path}")
    target_pcd = load_point_cloud(target_pcd_path)
    
    if method == "icp":
        print("\n使用ICP算法进行点云配准...")
        transform_matrix, fitness = register_point_clouds_icp(
            source_pcd, target_pcd,
            voxel_size=voxel_size,
            max_iterations=max_iterations,
            verbose=True
        )
        print(f"\n配准完成！变换矩阵:\n{transform_matrix}")
        print(f"配准质量分数: {fitness:.4f}")
        
    elif method == "camera":
        if source_calibration_path is None or target_calibration_path is None:
            raise ValueError("使用camera方法时需要提供两个相机标定文件")
        
        print("\n使用相机外参计算变换矩阵...")
        transform_matrix = compute_transform_from_cameras(
            source_calibration_path,
            target_calibration_path,
            reference_camera_name=reference_camera_name,
            use_multiple_cameras=use_multiple_cameras,
            common_camera_names=common_camera_names,
            use_array_geometry=use_array_geometry,
            source_is_json=source_is_json,
            target_is_json=target_is_json
        )
        print(f"\n变换矩阵:\n{transform_matrix}")
        fitness = None
        
    else:
        raise ValueError(f"未知的配准方法: {method}。请使用 'icp' 或 'camera'")
    
    # 应用变换
    print("\n应用变换到源点云...")
    transformed_pcd = apply_transform_to_point_cloud(source_pcd, transform_matrix)
    
    # 保存结果
    save_point_cloud(transformed_pcd, output_pcd_path)
    
    return transform_matrix, fitness


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="点云配准工具：将点云C对齐到点云D")
    parser.add_argument("--source_pcd", type=Path, required=True, help="源点云文件路径（点云C）")
    parser.add_argument("--target_pcd", type=Path, required=True, help="目标点云文件路径（点云D）")
    parser.add_argument("--output_pcd", type=Path, required=True, help="输出对齐后的点云文件路径")
    parser.add_argument("--method", type=str, default="icp", choices=["icp", "camera"],
                       help="配准方法：'icp'（自动配准）或 'camera'（使用相机外参）")
    parser.add_argument("--source_calibration", type=Path, default=None,
                       help="源相机阵列的标定文件（CSV或JSON，method='camera'时需要）")
    parser.add_argument("--target_calibration", type=Path, default=None,
                       help="目标相机阵列的标定文件（CSV或JSON，method='camera'时需要）")
    parser.add_argument("--source_is_json", action="store_true",
                       help="源标定文件为JSON格式（transforms.json）")
    parser.add_argument("--target_is_json", action="store_true",
                       help="目标标定文件为JSON格式（transforms.json）")
    parser.add_argument("--reference_camera", type=str, default="Cam127",
                       help="参考相机名称（method='camera'单相机方法时使用）")
    parser.add_argument("--use_multiple_cameras", action="store_true",
                       help="使用多相机方法计算变换（method='camera'时，更准确但需要公共相机）")
    parser.add_argument("--use_array_geometry", action="store_true",
                       help="使用几何结构方法（当没有公共相机时，自动启用）")
    parser.add_argument("--common_cameras", type=str, nargs="+", default=None,
                       help="公共相机名称列表（多相机方法，如不指定则自动查找）")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                       help="ICP下采样体素大小（米）")
    parser.add_argument("--max_iterations", type=int, default=30,
                       help="ICP最大迭代次数")
    
    args = parser.parse_args()
    
    register_and_align_point_clouds(
        source_pcd_path=args.source_pcd,
        target_pcd_path=args.target_pcd,
        output_pcd_path=args.output_pcd,
        method=args.method,
        source_calibration_path=args.source_calibration,
        target_calibration_path=args.target_calibration,
        reference_camera_name=args.reference_camera,
        use_multiple_cameras=args.use_multiple_cameras,
        use_array_geometry=args.use_array_geometry,
        common_camera_names=args.common_cameras,
        source_is_json=args.source_is_json,
        target_is_json=args.target_is_json,
        voxel_size=args.voxel_size,
        max_iterations=args.max_iterations
    )

