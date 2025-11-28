from __future__ import annotations

import json
import os.path as osp
from typing import Any, Dict

import numpy as np


def _build_intrinsics(frame: Dict[str, Any]) -> np.ndarray:
    fl_x = frame.get("fl_x")
    fl_y = frame.get("fl_y", fl_x)
    cx = frame.get("cx", frame.get("w", 0) / 2.0)
    cy = frame.get("cy", frame.get("h", 0) / 2.0)

    return np.array(
        [
            [fl_x, 0.0, cx],
            [0.0, fl_y, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _normalize_label(raw_label: Any) -> str:
    if raw_label is None:
        return ""
    if isinstance(raw_label, (int, float)):
        return f"{int(raw_label):02d}"

    raw_label = str(raw_label).strip()
    if raw_label.isdigit():
        return f"{int(raw_label):02d}"
    return raw_label


def parse_cameras(
    camera_path: str,
    coord_system: str = "opencv",
    normalize_scene: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Parse a nerfstudio-style transforms.json file into intrinsics/extrinsics."""

    if not osp.isfile(camera_path):
        raise FileNotFoundError(f"Camera file not found: {camera_path}")

    with open(camera_path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}

    frames = data.get("frames") or []
    if not frames:
        raise ValueError(f"No frames found in camera file: {camera_path}")

    coord_system = coord_system.lower()
    if coord_system not in {"opencv", "opengl"}:
        raise ValueError(f"Unsupported coord_system '{coord_system}'")

    cameras: Dict[str, Dict[str, Any]] = {}
    cam_centers = []

    for idx, frame in enumerate(frames):
        label = _normalize_label(
            frame.get("camera_label")
            or frame.get("camera_id")
            or frame.get("uid")
            or frame.get("img_id")
            or idx
        )
        pose = np.array(frame["transform_matrix"], dtype=np.float64)
        if coord_system == "opencv":
            cam_pose = pose
        else:
            convert = np.diag([1, -1, -1, 1]).astype(np.float64)
            cam_pose = pose @ convert

        cameras[label] = {
            "K": _build_intrinsics(frame),
            "pose": cam_pose,
            "width": frame.get("w"),
            "height": frame.get("h"),
            "distortion": {
                "k1": frame.get("k1"),
                "k2": frame.get("k2"),
                "p1": frame.get("p1"),
                "p2": frame.get("p2"),
                "k3": frame.get("k3"),
                "k4": frame.get("k4"),
            },
        }
        cam_centers.append(np.linalg.inv(cam_pose)[:3, 3])

    if normalize_scene and cam_centers:
        cam_centers = np.stack(cam_centers)
        center = cam_centers.mean(axis=0)
        radius = np.linalg.norm(cam_centers - center, axis=1).max()
        scale = 1.0 / max(radius, 1e-6)
        for meta in cameras.values():
            pose = meta["pose"].copy()
            pose[:3, 3] = (pose[:3, 3] - center) * scale
            meta["pose"] = pose

    return cameras

