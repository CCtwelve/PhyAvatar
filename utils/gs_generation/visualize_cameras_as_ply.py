import json
import numpy as np
import argparse
from pathlib import Path

def save_camera_to_ply(ply_path, vertices, edges, edge_colors):
    """
    将单个或多个相机的线框几何体（顶点、带颜色的边）保存为 PLY 文件。
    
    Args:
        ply_path (Path): 要保存的 .ply 文件路径。
        vertices (np.array): (N, 3) 形状的顶点坐标数组。
        edges (list): (K, 2) 形状的边索引列表，每个元素为 (idx1, idx2)。
        edge_colors (list): (K, 3) 形状的边颜色列表，每个元素为 (R, G, B)。
    """
    num_vertices = len(vertices)
    num_edges = len(edges)
    
    try:
        with open(ply_path, 'w') as f:
            # --- PLY Header ---
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            
            # 顶点定义
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            # 边定义 (带颜色)
            f.write(f"element edge {num_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            # --- PLY Data ---
            
            # 写入顶点
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
            # 写入边和颜色
            for i, edge in enumerate(edges):
                color = edge_colors[i]
                f.write(f"{edge[0]} {edge[1]} {color[0]} {color[1]} {color[2]}\n")
                
    except Exception as e:
        print(f"Error writing to {ply_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="将 transforms.json 中的相机姿态可视化为 单个PLY 文件")

    path = '/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48'
    # path = '/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/0023_06'
    # 假设的默认路径，基于我们之前的脚本
    name = "transforms"
    parser.add_argument(
        "--transforms_json", "-t", 
        type=str, 
        default=path+f"/{name}.json",
    )
    parser.add_argument(
        "--output_file", "-o", 
        type=str, 
        default=path +f'/{name}.ply',
        help="保存所有相机姿态的 *单个* PLY 文件的输出路径。默认: .../all_cameras.ply"
    )
    parser.add_argument(
        "--scale", 
        type=float, 
        default=0.2,
        help="相机三棱锥的可视化大小。默认: 0.1"
    )
    
    args = parser.parse_args()

    input_path = Path(args.transforms_json)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"错误: 找不到Inputs transforms.json 文件: {input_path}")
        return

    # 1. 定义相机的本地几何形状 (一个三棱锥 + 一条射线)
    # --- 关键修正: 匹配 Nerfstudio/OpenGL 坐标系 ( +Y向上, -Z向前 ) ---
    # --- 用户要求: 底面朝向物体 (-Z方向) ---
    s = args.scale
    scale_x = s * 0.8
    scale_y = s * 0.6 # +Y 向上
    scale_z_base = s * 1.0 # 底面深度
    scale_z_ray = s * 15 # 射线长度
    
    # v0: 锥顶 (相机中心)
    # v1-v4: 锥底 (图像平面)
    # v5: 视线射线的尖端
    local_vertices = np.array([
        [0, 0, 0],                                # v0 (锥顶, 相机中心)
        
        # --- 关键: 底座 (图像平面) 位于 Z 轴负方向 (相机前方, 朝向物体) ---
        [-scale_x, -scale_y, -scale_z_base],      # v1 (左下)
        [scale_x, -scale_y, -scale_z_base],       # v2 (右下)
        [scale_x, scale_y, -scale_z_base],        # v3 (右上)
        [-scale_x, scale_y, -scale_z_base],       # v4 (左上)
        
        # --- 关键: 视线射线 (v5) 同样指向 -Z 方向 (相机前方) ---
        [0, 0, -scale_z_ray]                      # v5 (视线射线尖端, 沿 -Z)
    ])
    
    # 3. 定义边 (9条边)
    # (这些是本地索引)
    local_edges = [
        (0, 5), # 视线射线 [关键]
        (0, 1), # 锥顶 -> 底座
        (0, 2), # 锥顶 -> 底座
        (0, 3), # 锥顶 -> 底座
        (0, 4), # 锥顶 -> 底座
        (1, 2), # 底座边
        (2, 3), # 底座边
        (3, 4), # 底座边
        (4, 1)  # 底座边
    ]
    
    # 4. 定义边的颜色
    COLOR_RAY = (255, 0, 0)     # 红色
    COLOR_FRAME = (255, 255, 255) # 白色
    
    local_edge_colors = [
        COLOR_RAY,
        COLOR_FRAME,
        COLOR_FRAME,
        COLOR_FRAME,
        COLOR_FRAME,
        COLOR_FRAME,
        COLOR_FRAME,
        COLOR_FRAME,
        COLOR_FRAME
    ]

    # 5. 加载 transforms.json
    print(f"正在加载相机姿态: {input_path}")
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载json文件失败: {e}")
        return

    # 6. 遍历所有 'frames' 并转换几何体
    num_cameras = len(data.get('frames', []))
    if num_cameras == 0:
        print("错误: 'transforms.json' 中未找到 'frames' 列表。")
        return
        
    print(f"正在聚合 {num_cameras} 个相机...")
    
    all_vertices_list = []
    all_edges = []
    all_edge_colors = []
    
    num_local_vertices = len(local_vertices) # 现在是 6

    for frame in data['frames']:
        
        current_vertex_offset = len(all_vertices_list) * num_local_vertices
        
        # 直接使用 json 中的 OpenGL 格式的 c2w 矩阵
        c2w = np.array(frame['transform_matrix'])
        
        # --- 关键步骤: 将本地 OpenGL 顶点变换到世界坐标系 ---
        
        # 1. 将本地顶点转为齐次坐标 (N, 4)
        local_vertices_hom = np.hstack((
            local_vertices, 
            np.ones((local_vertices.shape[0], 1))
        ))
        
        # 2. 应用 c2w 矩阵
        world_vertices_hom = (c2w @ local_vertices_hom.T).T
        
        # 3. 从齐次坐标转回 3D 坐标 (N, 3)
        world_vertices = world_vertices_hom[:, :3]
        
        # 7. 聚合几何数据
        all_vertices_list.append(world_vertices)
        
        # 添加偏移后的边索引
        for edge in local_edges:
            all_edges.append((
                edge[0] + current_vertex_offset,
                edge[1] + current_vertex_offset
            ))
        
        # 颜色不需要偏移
        all_edge_colors.extend(local_edge_colors)

    # 8. 循环结束后，保存单个聚合文件
    if not all_vertices_list:
        print("未处理任何相机。")
        return

    print(f"正在将 {num_cameras} 个相机合并到: {output_file} ...")
    
    all_vertices_np = np.concatenate(all_vertices_list, axis=0)
    
    save_camera_to_ply(
        output_file,
        all_vertices_np,
        all_edges,
        all_edge_colors
    )

    print(f"\n\033[92m转换成功!\033[0m")
    print(f"已将所有 {num_cameras} 个相机保存到 '{output_file}'。")
    print("您可以将此单个文件拖入 MeshLab 或 CloudCompare 查看。")

if __name__ == "__main__":
    main()