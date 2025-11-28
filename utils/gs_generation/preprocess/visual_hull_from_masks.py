#!/usr/bin/env python3
"""Construct a visual hull from multi-view masks and render extra masks.

Example:
    python scripts/preprocess/visual_hull_from_masks.py \
        --transforms_path /mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/transforms.json \
        --dataset_root /mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17 \
        --carve_views 00,01,02,03,04,17,15 \
        --frame_id 000000 \
        --render_views all \
        --output_dir /mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_17/visual_hull_masks
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


DEFAULT_CARVE_VIEWS = ("00", "01", "02", "03", "04")


@dataclass(frozen=True)
class Camera:
    label: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    w2c: np.ndarray  # (4, 4)
    c2w: np.ndarray  # (4, 4)
    mask_rel_path: Path
    forward_z_sign: float = field(init=False)

    def __post_init__(self) -> None:
        # Determine whether points in front of the camera have positive or negative z
        # in camera coordinates (nerfstudio uses -Z forward).
        forward_world = self.c2w[:3, 3] - self.c2w[:3, 2]
        forward_cam = self.w2c @ np.array([*forward_world, 1.0])
        sign = 1.0 if forward_cam[2] >= 0 else -1.0
        object.__setattr__(self, "forward_z_sign", sign)

    @property
    def camera_center(self) -> np.ndarray:
        return self.c2w[:3, 3]

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project Nx3 world points to pixel coordinates.

        Returns:
            valid_mask: bool array of shape (N,) for points inside view and z>0
            u: float array (N,)
            v: float array (N,)
        """
        rot = self.w2c[:3, :3]
        trans = self.w2c[:3, 3]
        points_cam = points_world @ rot.T + trans[None, :]
        z = points_cam[:, 2]
        if self.forward_z_sign > 0:
            depth = z
            denom = np.maximum(z, 1e-6)
        else:
            depth = -z
            denom = np.maximum(-z, 1e-6)
        valid = depth > 1e-6
        x_norm = points_cam[:, 0] / denom
        y_norm = points_cam[:, 1] / denom
        u = self.fx * x_norm + self.cx
        v = self.fy * y_norm + self.cy
        valid &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return valid, u, v


def parse_view_list(raw: str | None, default: Sequence[str]) -> Sequence[str]:
    if raw is None or raw.strip().lower() == "none":
        return []
    if raw.strip().lower() == "all":
        return default
    return tuple(v.strip() for v in raw.split(",") if v.strip())


def load_cameras(transforms_path: Path, dataset_root: Path, frame_id: str) -> dict[str, Camera]:
    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cameras: dict[str, Camera] = {}
    frames = meta.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {transforms_path}")

    for frame in frames:
        label = frame["camera_label"]
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        mask_rel = Path(frame["mask_path"])
        mask_rel = mask_rel.with_name(f"{frame_id}{mask_rel.suffix}")
        cameras[label] = Camera(
            label=label,
            width=int(frame["w"]),
            height=int(frame["h"]),
            fx=float(frame["fl_x"]),
            fy=float(frame["fl_y"]),
            cx=float(frame["cx"]),
            cy=float(frame["cy"]),
            c2w=c2w,
            w2c=w2c,
            mask_rel_path=mask_rel,
        )
    return cameras


def load_mask(mask_path: Path, threshold: float) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img, dtype=np.float32) / 255.0
    return mask >= threshold


def create_voxel_grid(bounds_min: Sequence[float], bounds_max: Sequence[float], resolution: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    linspaces = [np.linspace(bounds_min[i], bounds_max[i], resolution, dtype=np.float32) for i in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(*linspaces, indexing="ij")
    points_world = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return points_world, (grid_x, grid_y, grid_z)


def carve_visual_hull(
    points_world: np.ndarray,
    cameras: Sequence[Camera],
    masks: Sequence[np.ndarray],
) -> np.ndarray:
    if len(cameras) != len(masks):
        raise ValueError("Cameras and masks counts must match")
    occupancy = np.ones(points_world.shape[0], dtype=bool)
    for cam, mask in zip(cameras, masks):
        valid, u, v = cam.project(points_world)
        if not np.any(valid):
            continue
        u_idx = u[valid].astype(np.int32)
        v_idx = v[valid].astype(np.int32)
        mask_hits = mask[v_idx, u_idx]
        carve_mask = np.zeros_like(valid, dtype=bool)
        carve_mask[valid] = mask_hits
        occupancy &= carve_mask
        if not np.any(occupancy):
            break
    return occupancy


def render_masks_from_voxels(
    occupied_points: np.ndarray,
    cameras: Sequence[Camera],
    output_dir: Path,
    frame_id: str,
) -> None:
    if occupied_points.size == 0:
        raise RuntimeError("No occupied voxels to render")
    for cam in cameras:
        valid, u, v = cam.project(occupied_points)
        if not np.any(valid):
            continue
        z = (occupied_points @ cam.w2c[:3, :3].T + cam.w2c[:3, 3][None, :])[:, 2]
        point_depth = z if cam.forward_z_sign > 0 else -z
        point_depth = point_depth[valid]
        u_idx = u[valid].astype(np.int32)
        v_idx = v[valid].astype(np.int32)
        depth_img = np.full((cam.height, cam.width), np.inf, dtype=np.float32)
        depth_flat = depth_img.reshape(-1)
        flat_idx = v_idx * cam.width + u_idx
        np.minimum.at(depth_flat, flat_idx, point_depth)
        mask = (depth_img < np.inf).astype(np.uint8) * 255
        save_path = output_dir / cam.label / f"{frame_id}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(save_path)


def save_points_as_ply(points: np.ndarray, path: Path) -> None:
    """Write an (N, 3) array of points to a binary PLY file."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PLY export expects an (N, 3) array of points")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to little-endian float32 and ensure C-contiguous
    points = points.astype('<f4')  # '<f4' = little-endian float32
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)
    
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    
    with open(path, "wb") as f:
        # Write header as text (ASCII)
        header_bytes = "\n".join(header).encode("ascii") + b"\n"
        f.write(header_bytes)
        # Write binary vertex data (little-endian float32)
        # Each vertex is 3 floats = 12 bytes
        f.write(points.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual hull carving from multi-view masks.")
    parser.add_argument("--transforms_path", type=Path, required=True, help="Path to transforms.json.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Dataset root containing the fmasks directory.")
    parser.add_argument("--carve_views", type=str, default=",".join(DEFAULT_CARVE_VIEWS), help="Comma-separated camera labels used for carving.")
    parser.add_argument("--render_views", type=str, default="all", help="Comma-separated camera labels to render (or 'all').")
    parser.add_argument("--frame_id", type=str, default="000000", help="Frame identifier (e.g., 000000).")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Foreground threshold for mask binarization in [0,1].")
    parser.add_argument("--resolution", type=int, default=128, help="Number of voxels along each axis.")
    parser.add_argument("--bbox_min", type=str, default="-1.5,-1.0,-1.5", help="Comma-separated min bounds (x,y,z).")
    parser.add_argument("--bbox_max", type=str, default="1.5,2.5,1.5", help="Comma-separated max bounds (x,y,z).")
    parser.add_argument("--output_dir", type=Path, default=Path("./visual_hull_output"), help="Directory to store rendered masks and occupancy grid.")
    parser.add_argument("--store_occupancy", action="store_true", help="Save the occupancy grid as npz.")
    parser.add_argument(
        "--ply_path",
        type=Path,
        default=None,
        help="Optional path to save carved points as PLY (defaults to output_dir/visual_hull_<frame_id>.ply).",
    )

    args = parser.parse_args()

    bbox_min = tuple(float(v) for v in args.bbox_min.split(","))
    bbox_max = tuple(float(v) for v in args.bbox_max.split(","))

    cameras = load_cameras(args.transforms_path, args.dataset_root, args.frame_id)
    all_view_labels = tuple(sorted(cameras.keys()))

    carve_view_labels = parse_view_list(args.carve_views, DEFAULT_CARVE_VIEWS)
    missing = [v for v in carve_view_labels if v not in cameras]
    if missing:
        raise ValueError(f"Unknown carve views: {missing}. Available: {all_view_labels}")

    render_view_labels = parse_view_list(args.render_views, all_view_labels)
    if not render_view_labels:
        render_view_labels = all_view_labels
    missing = [v for v in render_view_labels if v not in cameras]
    if missing:
        raise ValueError(f"Unknown render views: {missing}")

    carve_cameras = [cameras[v] for v in carve_view_labels]
    render_cameras = [cameras[v] for v in render_view_labels]

    masks: list[np.ndarray] = []
    for cam in carve_cameras:
        mask_path = args.dataset_root / cam.mask_rel_path
        masks.append(load_mask(mask_path, args.mask_threshold))

    points_world, _ = create_voxel_grid(bbox_min, bbox_max, args.resolution)
    occupancy_flat = carve_visual_hull(points_world, carve_cameras, masks)
    occupied_points = points_world[occupancy_flat]

    if occupied_points.size == 0:
        raise RuntimeError("Visual hull is empty. Try adjusting bounds or threshold.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = args.ply_path or args.output_dir / f"visual_hull_{args.frame_id}.ply"
    save_points_as_ply(occupied_points, ply_path)

    if args.store_occupancy:
        np.savez_compressed(
            args.output_dir / f"occupancy_{args.frame_id}.npz",
            occupancy=occupancy_flat,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=args.resolution,
        )

    render_masks_from_voxels(occupied_points, render_cameras, args.output_dir, args.frame_id)

    print(f"Rendered masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()

