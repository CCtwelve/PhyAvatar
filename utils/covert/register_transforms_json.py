#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将source的transforms.json按照target的位置进行z轴反转
简单地将source的z坐标取反，使其与target对齐
"""

import json
import numpy as np
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation


def flip_z_axis_transforms_json(
    source_json_path: Path,
    target_json_path: Path,
    output_json_path: Path
):
    """
    将source的transforms.json进行z轴反转
    
    Args:
        source_json_path: 源transforms.json路径
        target_json_path: 目标transforms.json路径（用于参考ply_file_path等）
        output_json_path: 输出z轴反转后的transforms.json路径
    """
    print(f"读取源transforms.json: {source_json_path}")
    print(f"读取目标transforms.json: {target_json_path}")
    
    # 读取源JSON
    with open(source_json_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # 读取目标JSON（用于获取ply_file_path等元数据）
    with open(target_json_path, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    # Z轴取反矩阵
    flip_z = np.diag([1, 1, -1])
    
    # 应用z轴反转到每个frame的transform_matrix
    transformed_frames = []
    for frame in source_data.get('frames', []):
        # 获取原始c2w矩阵
        original_c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        
        # 提取旋转和平移
        R = original_c2w[:3, :3]
        t = original_c2w[:3, 3]
        
        # 对平移向量应用z轴取反
        t_flipped = flip_z @ t
        
        # 对旋转矩阵应用z轴取反：R' = flip_z @ R @ flip_z^T
        R_flipped = flip_z @ R @ flip_z.T
        
        # 构建新的c2w矩阵
        transformed_c2w = np.eye(4, dtype=np.float32)
        transformed_c2w[:3, :3] = R_flipped
        transformed_c2w[:3, 3] = t_flipped
        
        # 创建新的frame
        new_frame = frame.copy()
        new_frame['transform_matrix'] = transformed_c2w.tolist()
        transformed_frames.append(new_frame)
    
    # 构建输出数据
    output_data = {
        "ply_file_path": "sparse_pcd.ply",
        'frames': transformed_frames
    }
    
    # 如果目标JSON有ply_file_path，也复制过来
    if 'ply_file_path' in target_data:
        output_data['ply_file_path'] = target_data['ply_file_path']
    
    # 保存结果
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nz轴反转后的transforms.json已保存到: {output_json_path}")
    print(f"共处理了 {len(transformed_frames)} 个相机")
    
    # 打印一些统计信息
    source_positions = np.array([np.array(frame['transform_matrix'])[:3, 3] for frame in source_data.get('frames', [])])
    target_positions = np.array([np.array(frame['transform_matrix'])[:3, 3] for frame in target_data.get('frames', [])])
    transformed_positions = np.array([np.array(frame['transform_matrix'])[:3, 3] for frame in transformed_frames])
    
    source_centroid = np.mean(source_positions, axis=0)
    target_centroid = np.mean(target_positions, axis=0)
    transformed_centroid = np.mean(transformed_positions, axis=0)
    
    print(f"\n统计信息:")
    print(f"  源相机中心: ({source_centroid[0]:.4f}, {source_centroid[1]:.4f}, {source_centroid[2]:.4f})")
    print(f"  目标相机中心: ({target_centroid[0]:.4f}, {target_centroid[1]:.4f}, {target_centroid[2]:.4f})")
    print(f"  反转后源相机中心: ({transformed_centroid[0]:.4f}, {transformed_centroid[1]:.4f}, {transformed_centroid[2]:.4f})")
    print(f"  z轴反转: 源z={source_centroid[2]:.4f} -> 反转后z={transformed_centroid[2]:.4f} (目标z={target_centroid[2]:.4f})")


def extract_camera_poses_from_json(json_path: Path):
    """
    从transforms.json提取相机位置和朝向
    
    Returns:
        positions: (N, 3) 相机位置数组
        directions: (N, 3) 相机朝向数组（相机看向的方向，即c2w的-z轴）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    positions = []
    directions = []
    
    for frame in data.get('frames', []):
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        
        # 相机位置（c2w的平移部分）
        position = c2w[:3, 3]
        positions.append(position)
        
        # 相机朝向（c2w的-z轴方向，即相机看向的方向）
        # c2w的第三列（索引2）是-z轴在世界坐标系中的方向
        forward = -c2w[:3, 2]  # 相机看向的方向
        forward = forward / (np.linalg.norm(forward) + 1e-8)  # 归一化
        directions.append(forward)
    
    return np.array(positions), np.array(directions)


def create_camera_visualization(
    positions: np.ndarray,
    directions: np.ndarray,
    color: tuple,
    scale: float = 0.1,
    arrow_length: float = 0.3
):
    """
    为相机阵列创建可视化几何体（三棱锥，底面朝向物体）
    
    Args:
        positions: (N, 3) 相机位置数组
        directions: (N, 3) 相机朝向数组（归一化，相机看向的方向）
        color: (R, G, B) 颜色元组，值范围0-255
        scale: 三棱锥底面大小
        arrow_length: 棱锥高度（从顶点到底面的距离）
    
    Returns:
        vertices: (M, 3) 所有顶点坐标
        edges: (K, 2) 边索引列表
        edge_colors: (K, 3) 边颜色列表
        vertex_colors: (M, 3) 顶点颜色列表
        faces: (F, 3) 面索引列表（每个面由3个顶点索引组成）
        face_colors: (F, 3) 面颜色列表
    """
    all_vertices = []
    all_edges = []
    all_edge_colors = []
    all_vertex_colors = []
    all_faces = []
    all_face_colors = []
    
    vertex_offset = 0
    
    for i, (pos, dir_vec) in enumerate(zip(positions, directions)):
        # 三棱锥：顶点在相机位置（稍微向后），底面在相机前方（朝向物体）
        # dir_vec 是相机看向的方向（从相机指向物体）
        
        # 计算垂直于相机朝向的平面（用于放置底面）
        # 选择一个参考向量来构建正交基
        if abs(dir_vec[2]) < 0.9:
            ref_vec = np.array([0, 0, 1])  # 使用z轴作为参考
        else:
            ref_vec = np.array([1, 0, 0])  # 如果z轴太接近，使用x轴
        
        # 构建正交基：dir_vec（相机朝向），u（底面方向1），v（底面方向2）
        u = np.cross(dir_vec, ref_vec)
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(dir_vec, u)
        v = v / (np.linalg.norm(v) + 1e-8)
        
        # 三棱锥参数
        base_distance = arrow_length  # 底面距离顶点的距离（使用arrow_length作为高度）
        base_radius = scale  # 底面等边三角形的外接圆半径
        
        # 顶点位置（在相机位置，稍微向后偏移一点）
        apex = pos - dir_vec * base_distance * 0.1
        
        # 底面中心（在相机前方，朝向物体）
        base_center = pos + dir_vec * base_distance
        
        # 底面三个顶点（等边三角形）
        # 在垂直于dir_vec的平面上，以base_center为中心
        angle_120 = 2 * np.pi / 3
        base_vertices = []
        for j in range(3):
            angle = j * angle_120
            # 在u-v平面上旋转
            offset = base_radius * (np.cos(angle) * u + np.sin(angle) * v)
            base_vertex = base_center + offset
            base_vertices.append(base_vertex)
        
        # 三棱锥的4个顶点：1个顶点 + 3个底面顶点
        pyramid_vertices = np.vstack([apex.reshape(1, 3), np.array(base_vertices)])
        
        # 顶点索引：0=顶点，1,2,3=底面三个顶点
        # 三棱锥的6条边：
        # - 3条从顶点到底面顶点的边：(0,1), (0,2), (0,3)
        # - 3条底面边：(1,2), (2,3), (3,1)
        pyramid_edges = [
            (0, 1), (0, 2), (0, 3),  # 从顶点到底面
            (1, 2), (2, 3), (3, 1)   # 底面三角形
        ]
        
        # 三棱锥的4个面：
        # - 3个侧面：每个侧面由顶点和底面的两个相邻顶点组成
        # - 1个底面：由三个底面顶点组成（朝向物体，法向量沿dir_vec方向）
        # 确保底面面的顶点顺序使得法向量指向dir_vec方向（朝向物体）
        # 计算底面的法向量来验证方向
        base_v1 = base_vertices[0] - base_center
        base_v2 = base_vertices[1] - base_center
        base_normal = np.cross(base_v1, base_v2)
        base_normal = base_normal / (np.linalg.norm(base_normal) + 1e-8)
        
        # 如果法向量与dir_vec方向相反，则反转顶点顺序
        if np.dot(base_normal, dir_vec) < 0:
            # 反转底面顶点顺序，使法向量指向dir_vec方向
            base_face = (1, 3, 2)  # 反转顺序：(1, 2, 3) -> (1, 3, 2)
        else:
            base_face = (1, 2, 3)  # 保持原顺序
        
        pyramid_faces = [
            (0, 1, 2),  # 侧面1：顶点、底面顶点1、底面顶点2
            (0, 2, 3),  # 侧面2：顶点、底面顶点2、底面顶点3
            (0, 3, 1),  # 侧面3：顶点、底面顶点3、底面顶点1
            base_face   # 底面：底面三角形（法向量朝向物体，沿dir_vec方向）
        ]
        
        # 添加偏移
        camera_edges = [(e[0] + vertex_offset, e[1] + vertex_offset) for e in pyramid_edges]
        all_edges.extend(camera_edges)
        
        camera_faces = [(f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset) for f in pyramid_faces]
        all_faces.extend(camera_faces)
        
        # 顶点颜色（所有顶点用相同颜色）
        camera_vertex_colors = np.tile(np.array(color), (len(pyramid_vertices), 1))
        all_vertex_colors.append(camera_vertex_colors)
        
        # 边颜色（所有边用相同颜色）
        camera_edge_colors = [color] * len(pyramid_edges)
        all_edge_colors.extend(camera_edge_colors)
        
        # 面颜色（所有面用相同颜色）
        camera_face_colors = [color] * len(pyramid_faces)
        all_face_colors.extend(camera_face_colors)
        
        all_vertices.append(pyramid_vertices)
        vertex_offset += len(pyramid_vertices)
    
    # 合并所有数据
    vertices = np.vstack(all_vertices) if all_vertices else np.array([]).reshape(0, 3)
    vertex_colors = np.vstack(all_vertex_colors) if all_vertex_colors else np.array([]).reshape(0, 3)
    
    return vertices, all_edges, all_edge_colors, vertex_colors, all_faces, all_face_colors


def save_cameras_to_ply(
    ply_path: Path,
    vertices: np.ndarray,
    edges: list,
    edge_colors: list,
    vertex_colors: np.ndarray = None,
    faces: list = None,
    face_colors: list = None
):
    """
    将相机可视化（顶点、边、面、颜色）保存为 PLY 文件
    
    Args:
        ply_path: 输出PLY文件路径
        vertices: (N, 3) 顶点坐标数组
        edges: (K, 2) 边索引列表
        edge_colors: (K, 3) 边颜色列表，每个元素为 (R, G, B) 0-255
        vertex_colors: (N, 3) 可选，顶点颜色数组，每个元素为 (R, G, B) 0-255
        faces: (F, 3) 可选，面索引列表，每个元素为 (v1, v2, v3) 三个顶点索引
        face_colors: (F, 3) 可选，面颜色列表，每个元素为 (R, G, B) 0-255
    """
    num_vertices = len(vertices)
    num_edges = len(edges)
    num_faces = len(faces) if faces is not None else 0
    
    try:
        with open(ply_path, 'w') as f:
            # PLY Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            
            # 顶点定义
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if vertex_colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            # 边定义（带颜色）
            if num_edges > 0:
                f.write(f"element edge {num_edges}\n")
                f.write("property int vertex1\n")
                f.write("property int vertex2\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            # 面定义（带颜色）
            if num_faces > 0:
                f.write(f"element face {num_faces}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            # 写入顶点
            for i, v in enumerate(vertices):
                if vertex_colors is not None:
                    color = vertex_colors[i]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
                else:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # 写入边和颜色
            for i, edge in enumerate(edges):
                color = edge_colors[i]
                f.write(f"{edge[0]} {edge[1]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
            
            # 写入面和颜色
            if faces is not None and face_colors is not None:
                for i, face in enumerate(faces):
                    color = face_colors[i]
                    # PLY格式：面由顶点数量（3）和三个顶点索引组成
                    f.write(f"3 {face[0]} {face[1]} {face[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
                
    except Exception as e:
        print(f"保存PLY文件时出错: {e}")
        raise


def visualize_cameras_to_ply(
    source_json_path: Path,
    target_json_path: Path,
    flipped_json_path: Path,
    output_ply_path: Path,
    scale: float = 0.1,
    arrow_length: float = 0.3
):
    """
    将三个transforms.json的相机位置和朝向分别保存到三个独立的PLY文件
    
    Args:
        source_json_path: 源transforms.json路径（原始坐标系）
        target_json_path: 目标transforms.json路径（用于参考）
        flipped_json_path: z轴反转后的transforms.json路径
        output_ply_path: 输出PLY文件路径（将作为目录，在其中生成三个PLY文件）
        scale: 相机位置点的可视化大小
        arrow_length: 朝向箭头的长度
    """
    print(f"\n开始提取相机位置和朝向...")
    
    # 定义颜色（RGB，0-255）
    COLOR_SOURCE = (255, 0, 0)      # 红色 - 源相机阵列（原始坐标系）
    COLOR_TARGET = (0, 0, 255)      # 蓝色 - 目标相机阵列
    COLOR_FLIPPED = (0, 255, 0)     # 绿色 - z轴反转后的源相机阵列
    
    # 如果output_ply_path是文件路径，则使用其父目录；如果是目录，直接使用
    if output_ply_path.suffix == '.ply':
        output_ply_dir = output_ply_path.parent
    else:
        output_ply_dir = output_ply_path
    
    # 确保输出目录存在
    output_ply_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 源相机阵列（原始坐标系）- 红色
    if source_json_path.exists():
        print(f"  读取源相机阵列: {source_json_path}")
        source_positions, source_directions = extract_camera_poses_from_json(source_json_path)
        source_vertices, source_edges, source_edge_colors, source_vertex_colors, source_faces, source_face_colors = create_camera_visualization(
            source_positions, source_directions, COLOR_SOURCE, scale, arrow_length
        )
        source_ply_path = output_ply_dir / "cameras_source.ply"
        print(f"  保存源相机阵列到: {source_ply_path}")
        save_cameras_to_ply(
            source_ply_path,
            source_vertices,
            source_edges,
            source_edge_colors,
            source_vertex_colors,
            source_faces,
            source_face_colors
        )
        print(f"    成功保存 {len(source_vertices)} 个顶点、{len(source_edges)} 条边和 {len(source_faces)} 个面（红色）")
    else:
        print(f"  警告: 源相机阵列文件不存在: {source_json_path}")
    
    # 2. 目标相机阵列 - 蓝色
    if target_json_path.exists():
        print(f"  读取目标相机阵列: {target_json_path}")
        target_positions, target_directions = extract_camera_poses_from_json(target_json_path)
        target_vertices, target_edges, target_edge_colors, target_vertex_colors, target_faces, target_face_colors = create_camera_visualization(
            target_positions, target_directions, COLOR_TARGET, scale, arrow_length
        )
        target_ply_path = output_ply_dir / "cameras_target.ply"
        print(f"  保存目标相机阵列到: {target_ply_path}")
        save_cameras_to_ply(
            target_ply_path,
            target_vertices,
            target_edges,
            target_edge_colors,
            target_vertex_colors,
            target_faces,
            target_face_colors
        )
        print(f"    成功保存 {len(target_vertices)} 个顶点、{len(target_edges)} 条边和 {len(target_faces)} 个面（蓝色）")
    else:
        print(f"  警告: 目标相机阵列文件不存在: {target_json_path}")
    
    # 3. z轴反转后的源相机阵列 - 绿色
    if flipped_json_path.exists():
        print(f"  读取z轴反转后的相机阵列: {flipped_json_path}")
        flipped_positions, flipped_directions = extract_camera_poses_from_json(flipped_json_path)
        flipped_vertices, flipped_edges, flipped_edge_colors, flipped_vertex_colors, flipped_faces, flipped_face_colors = create_camera_visualization(
            flipped_positions, flipped_directions, COLOR_FLIPPED, scale, arrow_length
        )
        flipped_ply_path = output_ply_dir / "cameras_flipped.ply"
        print(f"  保存z轴反转后的相机阵列到: {flipped_ply_path}")
        save_cameras_to_ply(
            flipped_ply_path,
            flipped_vertices,
            flipped_edges,
            flipped_edge_colors,
            flipped_vertex_colors,
            flipped_faces,
            flipped_face_colors
        )
        print(f"    成功保存 {len(flipped_vertices)} 个顶点、{len(flipped_edges)} 条边和 {len(flipped_faces)} 个面（绿色）")
    else:
        print(f"  警告: z轴反转后的相机阵列文件不存在: {flipped_json_path}")
    
    print(f"\n所有PLY文件已保存到: {output_ply_dir}")
    print(f"文件说明:")
    print(f"  - cameras_source.ply: 源相机阵列（原始坐标系，红色）")
    print(f"  - cameras_target.ply: 目标相机阵列（蓝色）")
    print(f"  - cameras_flipped.ply: z轴反转后的源相机阵列（绿色）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将source的transforms.json按照target的位置进行z轴反转"
    )
    parser.add_argument(
        "--source_json", "-s",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/0013_01/transforms.json"),
        help="源transforms.json路径"
    )
    parser.add_argument(
        "--target_json", "-t",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/transforms_.json"),
        help="目标transforms.json路径（用于参考ply_file_path等）"
    )
    parser.add_argument(
        "--output_json", "-o",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/transforms_registered_.json"),
    )
    parser.add_argument(
        "--output_ply",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17"),
        help="输出PLY文件目录或路径（用于可视化相机位置和朝向）。如果指定，将在该目录下生成三个独立的PLY文件：cameras_source.ply（红色）、cameras_target.ply（蓝色）、cameras_flipped.ply（绿色）"
    )
    parser.add_argument(
        "--ply_scale",
        type=float,
        default=0.1,
        help="PLY文件中相机位置点的可视化大小（默认: 0.1）"
    )
    parser.add_argument(
        "--ply_arrow_length",
        type=float,
        default=0.3,
        help="PLY文件中相机朝向箭头的长度（默认: 0.3）"
    )
    args = parser.parse_args()
    
    # 执行z轴反转
    flip_z_axis_transforms_json(
        source_json_path=args.source_json,
        target_json_path=args.target_json,
        output_json_path=args.output_json
    )
    
    # 如果指定了输出PLY路径，则保存相机可视化
    if args.output_ply is not None:
        print("\n" + "="*60)
        print("生成相机可视化PLY文件...")
        print("="*60)
        visualize_cameras_to_ply(
            source_json_path=args.source_json,
            target_json_path=args.target_json,
            flipped_json_path=args.output_json,
            output_ply_path=args.output_ply,
            scale=args.ply_scale,
            arrow_length=args.ply_arrow_length
        )

