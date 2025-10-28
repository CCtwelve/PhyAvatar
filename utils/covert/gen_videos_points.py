import open3d as o3d
import numpy as np
import cv2
import argparse
import math
from pathlib import Path

def create_orbiting_video(
    point_cloud_path: str,
    output_video_path: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: int = 10,
    up_axis: str = 'Z',
    orbit_radius_factor: float = 1.5,
    background_color: tuple = (0.1, 0.1, 0.1, 1.0),
    up_offset: float = 0.0
):
    """
    加载点云，将其按需平移，并让一个水平相机围绕原点拍摄视频。
    """
    print(f"1. 正在加载点云文件: {point_cloud_path}")
    try:
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        if not pcd.has_points():
            print("错误：点云为空或加载失败。")
            return
    except Exception as e:
        print(f"错误：无法读取点云文件。{e}")
        return

    print("   正在清理点云，移除离群点...")
    cleaned_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"   清理完毕。原始点数: {len(pcd.points)}, 清理后点数: {len(cleaned_pcd.points)}")
    
    print("2. 正在自动计算相机初始视点和轨道参数...")

    aabb = cleaned_pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    
    # 将点云的几何中心平移到世界原点
    cleaned_pcd.translate(-center, relative=False)
    
    up_vector_map = {'X': [1, 0, 0], 'Y': [0, 1, 0], 'Z': [0, 0, 1]}
    if up_axis.upper() not in up_vector_map:
        print(f"警告：无效的 up_axis '{up_axis}'。将使用 'Z' 作为默认值。")
        up_axis = 'Z'
    up = up_vector_map[up_axis.upper()]

    # 根据up_offset进行额外平移
    if up_offset != 0.0:
        up_translation_vector = np.array(up) * up_offset
        cleaned_pcd.translate(up_translation_vector, relative=True)
        print(f"   点云已沿 {up_axis} 轴额外平移 {up_offset} 个单位。")
    else:
        print(f"   点云已居中于原点。")

    max_extent = np.max(aabb.get_extent())
    radius = max_extent * orbit_radius_factor

    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    except Exception as e:
        print(f"\n错误: 无法初始化 OffscreenRenderer。\n原始错误: {e}\n")
        return

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0

    renderer.scene.add_geometry("pcd", cleaned_pcd, mat)
    renderer.scene.set_background(np.asarray(background_color))

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = duration * fps
    print(f"3. 准备就绪，将生成一个 {duration} 秒, {fps} FPS 的视频 ({total_frames} 帧)...")

    if up_axis.upper() == 'Z':
        orbit_axis1, orbit_axis2 = np.array([1, 0, 0]), np.array([0, 1, 0])
    elif up_axis.upper() == 'Y':
        orbit_axis1, orbit_axis2 = np.array([1, 0, 0]), np.array([0, 0, 1])
    else:
        orbit_axis1, orbit_axis2 = np.array([0, 1, 0]), np.array([0, 0, 1])

    # 定义相机的固定目标点和轨道中心
    orbit_center = np.array([0., 0., 0.])
        
    for i in range(total_frames):
        angle = (i / total_frames) * 2 * math.pi
        
        offset = radius * math.cos(angle) * orbit_axis1 + radius * math.sin(angle) * orbit_axis2
        
        # 相机轨道始终围绕世界原点
        camera_position = orbit_center + offset

        # --- 关键修正: 相机始终看向世界原点，不再抬头或低头 ---
        renderer.scene.camera.look_at(orbit_center, camera_position, up)

        image = renderer.render_to_image()
        frame = np.asarray(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video_writer.write(frame_bgr)
        
        if (i + 1) % fps == 0:
            print(f"   已处理 {i + 1} / {total_frames} 帧...")

    print(f"4. 视频生成完毕！已保存至: {output_video_path}")
    video_writer.release()

def main():
    parser = argparse.ArgumentParser(description="在无头环境中根据点云自动生成一个环绕拍摄的视频。")
    
    # --- 默认路径已根据您的要求设置为 .../3D.ply ---
    parser.add_argument("--point_cloud", type=Path, 
                        default=Path("/mnt/cvda/cvda_phava/code/Han/PhyAvatar/ActorsHQ-for-Gaussian-Garments/gs2mesh/data/custom/Actor01_Sequence1/sparse/0/3D.ply"), 
                        help="输入的点云文件路径")
    parser.add_argument("--output", "-o", type=Path, 
                        default=Path("/mnt/cvda/cvda_phava/code/Han/PhyAvatar/result/circle_render/output_video_points.mp4"), 
                        help="输出的视频文件路径")
    
    parser.add_argument("--width", type=int, default=1920, help="视频分辨率的宽度 (默认: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="视频分辨率的高度 (默认: 1080)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率 (默认: 30)")
    parser.add_argument("--duration", type=int, default=10, help="视频时长（秒） (默认: 10)")
    parser.add_argument("--up", type=str, default='Y', choices=['X', 'Y', 'Z'], help="定义点云的“向上”轴 (默认: Y)")
    parser.add_argument("--radius_factor", type=float, default=1.5, help="轨道半径系数，数值越小相机越近 (默认: 1.5)")
    parser.add_argument("--up_offset", type=float, default=0.5, help="沿指定的 'up' 轴平移点云的距离 (默认: 0.0)")
    
    args = parser.parse_args()
    
    if not args.point_cloud.exists():
        print(f"错误: 输入文件不存在 -> {args.point_cloud}")
        return

    create_orbiting_video(
        point_cloud_path=str(args.point_cloud),
        output_video_path=str(args.output),
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        up_axis=args.up,
        orbit_radius_factor=args.radius_factor,
        up_offset=args.up_offset
    )

if __name__ == "__main__":
    main()