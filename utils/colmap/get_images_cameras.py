#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Set

from scipy.spatial.transform import Rotation

from utils.covert.camera_data import CameraData, read_calibration_csv
import os


def get_image_files(folder_path: Path) -> Set[str]:
    """获取文件夹下所有图片文件名(不含路径)"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    return {
        filename.split('.')[0] for filename in os.listdir(folder_path)
    }


def export_as_colmap(cameras: List[CameraData], output_folder: Path, existing_images: Set[str]) -> None:
    camera_lines = []
    image_lines = []

    image_id=1
    for camera_id, camera in enumerate(cameras):
        # 只处理存在于图片文件夹中的相机/图像
        if camera.name not in existing_images:
            continue

        world_to_camera = Rotation.from_rotvec(-camera.rotation_axisangle)
        quat = world_to_camera.as_quat()
        tvec = -world_to_camera.as_matrix() @ camera.translation

        fx, fy, cx, cy = camera.fx_pixel, camera.fy_pixel, camera.cx_pixel, camera.cy_pixel
        # camera_lines.append(
        #     f"{camera_id+1} PINHOLE {camera.width} {camera.height} {fx} {fy} {cx} {cy}"
        # )

        camera_lines.append(
            f"{image_id} PINHOLE {camera.width} {camera.height} {fx} {fy} {cx} {cy}"
        )
        x, y, z, w = tuple(quat)
        tx, ty, tz = tuple(tvec)
        # image_lines.append(
        #     f"{image_id} {w} {x} {y} {z} {tx} {ty} {tz} {camera_id+1} {camera.name}.jpg" +'\n'
        # )
        image_lines.append(
            f"{image_id} {w} {x} {y} {z} {tx} {ty} {tz} {image_id} {camera.name}.jpg" +'\n'
        )
        image_id = image_id+1
    # 写入cameras.txt
    with open(output_folder / "cameras.txt", "w") as f:
        # f.write("# Camera list with one line of data per camera:\n")
        # f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # f.write("# Number of cameras: {}\n".format(len(camera_lines)))
        f.write("\n".join(camera_lines))

    # 写入images.txt
    with open(output_folder / "images.txt", "w") as f:
        # f.write("# Image list with two lines of data per image:\n")
        # f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        # f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        # f.write("# Number of images: {}, mean observations per image: 0\n".format(len(image_lines)))
        f.write("\n".join(image_lines))
        f.write("\n")
        # 为每个图像添加一个空行作为POINTS2D占位符
        # f.write("\n" * len(image_lines))

    # 写入空的points3D.txt
    with open(output_folder / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export camera calibration data to COLMAP format, "
                    "filtering by existing images."
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/calibration.csv"),
        help="Path to input calibration CSV file"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("/mnt/cvda/cvda_phava/code/Han/3DGS/colmap/data/cam_txt_17"),
        help="Output directory for COLMAP files"
    )
    parser.add_argument(
        "--image_path", type=Path, default=Path("/mnt/cvda/cvda_phava/code/Han/3DGS/colmap/data/multipleview/0_17"),
        help="Path to folder containing images to filter by"
    )
    args = parser.parse_args()

    # 获取存在的图片文件名集合
    existing_images = get_image_files(args.image_path)
    print( existing_images )
    print(f"Found {len(existing_images)} images in {args.image_path}")

    # 读取相机校准数据
    cameras = read_calibration_csv(args.csv)
    print(f"Read {len(cameras)} cameras from {args.csv}")

    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 导出过滤后的COLMAP格式文件
    export_as_colmap(cameras, args.output_dir, existing_images)
    print(f"Exported COLMAP files to {args.output_dir}")


if __name__ == "__main__":
    main()