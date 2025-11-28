import json
import os
from typing import Iterable, Sequence

import cv2
import fire
import numpy as np

from easyvolcap.utils.parallel_utils import parallel_execution
from sapiens.lite.demo.classes_and_palettes import (
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
    BLUE,
)

INVALID = -1e6


def score_to_color(rgb, score, low=0.5, high=0.9):
    score = np.clip(score, low, high)
    norm_score = (score - low) / (high - low)
    rgb = np.array(rgb, dtype=np.float32) * norm_score
    rgb = np.round(rgb, decimals=0).astype(np.uint8).tolist()
    return rgb


PLY_DTYPE_MAP = {
    "double": "<f8",
    "float": "<f4",
    "float32": "<f4",
    "float64": "<f8",
    "int": "<i4",
    "int32": "<i4",
    "int16": "<i2",
    "int8": "<i1",
    "uchar": "u1",
    "uint8": "u1",
}


def _load_ply_vertices(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header_ended = False
        vertex_count = 0
        vertex_properties: list[str] = []
        dtype_sequence: list[str] = []
        while not header_ended:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                header_ended = True
                break
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                _parts = line.split()
                dtype_string = PLY_DTYPE_MAP.get(_parts[1])
                if dtype_string is None:
                    raise ValueError(f"Unsupported PLY dtype {_parts[1]} in {path}")
                dtype_sequence.append(dtype_string)
                vertex_properties.append(_parts[-1])

        if vertex_count == 0:
            raise ValueError(f"No vertex element found in {path}")
        if len(vertex_properties) < 3:
            raise ValueError("PLY file must contain at least x,y,z properties")

        dtype = np.dtype([(name, dt) for name, dt in zip(vertex_properties, dtype_sequence)])
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1)
        return xyz.astype(np.float32)


def _convert_w2c_coordinate_system(
    w2c: np.ndarray, coord_convention: str
) -> np.ndarray:
    coord = coord_convention.lower()
    if coord == "opencv":
        return w2c
    if coord == "opengl":
        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
        return gl_to_cv @ w2c
    raise ValueError(f"Unsupported camera coordinate convention: {coord_convention}")


def _load_cameras(
    camera_path: str, spa_labels: Sequence[int] | None, coord_convention: str
) -> list[dict]:
    with open(camera_path, "r") as f:
        meta = json.load(f)
    if spa_labels is None:
        spa_filter = None
    else:
        spa_filter = {f"{spa:02d}" for spa in spa_labels}
    cameras = []
    for frame in meta["frames"]:
        label = frame["camera_label"]
        if spa_filter is not None and label not in spa_filter:
            continue
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        w2c = _convert_w2c_coordinate_system(w2c, coord_convention)
        cameras.append(
            {
                "label": label,
                "w": int(frame["w"]),
                "h": int(frame["h"]),
                "fx": float(frame["fl_x"]),
                "fy": float(frame["fl_y"]),
                "cx": float(frame["cx"]),
                "cy": float(frame["cy"]),
                "w2c": w2c,
            }
        )
    if not cameras:
        raise ValueError("No cameras loaded (check spa_labels filter)")
    return cameras


def _clip_line_to_bounds(p1: np.ndarray, p2: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Clip a line segment to image bounds using Cohen-Sutherland algorithm.
    Returns (clipped_p1, clipped_p2) if the line intersects the image bounds, None otherwise.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    
    # Compute region codes for both endpoints
    def compute_code(x, y):
        code = 0
        if x < 0:
            code |= 1  # left
        elif x >= w:
            code |= 2  # right
        if y < 0:
            code |= 4  # bottom
        elif y >= h:
            code |= 8  # top
        return code
    
    # Maximum iterations to prevent infinite loops
    max_iter = 10
    iter_count = 0
    
    while iter_count < max_iter:
        iter_count += 1
        code1 = compute_code(x1, y1)
        code2 = compute_code(x2, y2)
        
        # Trivially accept: both endpoints inside
        if code1 == 0 and code2 == 0:
            return np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)
        
        # Trivially reject: both endpoints on same side of boundary
        if code1 & code2 != 0:
            return None
        
        # Pick an endpoint that is outside
        outcode = code1 if code1 != 0 else code2
        
        # Find intersection point
        dx = x2 - x1
        dy = y2 - y1
        
        if outcode & 8:  # top
            x = x1 + dx * (h - 1 - y1) / dy if abs(dy) > 1e-10 else x1
            y = h - 1
        elif outcode & 4:  # bottom
            x = x1 + dx * (-y1) / dy if abs(dy) > 1e-10 else x1
            y = 0
        elif outcode & 2:  # right
            y = y1 + dy * (w - 1 - x1) / dx if abs(dx) > 1e-10 else y1
            x = w - 1
        elif outcode & 1:  # left
            y = y1 + dy * (-x1) / dx if abs(dx) > 1e-10 else y1
            x = 0
        
        # Replace the point outside with intersection point
        if outcode == code1:
            x1, y1 = x, y
        else:
            x2, y2 = x, y
    
    # If we exceed max iterations, return None (shouldn't happen normally)
    return None


def _project_points(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    
    # Check for invalid points (points that are INVALID or contain INVALID values)
    # A point is invalid if any of its coordinates is <= INVALID + 1e3
    point_valid = np.all(points > INVALID + 1e3, axis=1)
    
    keypoints = np.full((num_points, 2), INVALID, dtype=np.float32)
    scores = np.zeros(num_points, dtype=np.float32)
    depths = np.full(num_points, INVALID, dtype=np.float32)
    
    if not point_valid.any():
        return keypoints, scores, depths
    
    # Only project valid points
    valid_points = points[point_valid]
    num_valid = valid_points.shape[0]
    homog = np.concatenate([valid_points, np.ones((num_valid, 1), dtype=np.float32)], axis=1)
    cam_pts = (camera["w2c"] @ homog.T).T[:, :3]

    zs = cam_pts[:, 2]
    depth_valid = zs > 1e-6

    xs = cam_pts[:, 0]
    ys = cam_pts[:, 1]

    if depth_valid.any():
        valid_indices_relative = np.where(depth_valid)[0]
        valid_indices_absolute = np.where(point_valid)[0][valid_indices_relative]
        
        us = camera["fx"] * (xs[depth_valid] / zs[depth_valid]) + camera["cx"]
        vs = camera["fy"] * (ys[depth_valid] / zs[depth_valid]) + camera["cy"]

        # Don't clip coordinates - keep original projection results
        # Points outside image bounds will be filtered during rendering
        keypoints[valid_indices_absolute, 0] = us
        keypoints[valid_indices_absolute, 1] = vs
        scores[valid_indices_absolute] = 1.0
        depths[valid_indices_absolute] = zs[depth_valid]
    return keypoints, scores, depths


def _render_skeleton(
    keypoints: np.ndarray,
    scores: np.ndarray,
    depths: np.ndarray,
    out_path: str,
    canvas_shape: tuple[int, int],
    low_thr: float,
    high_thr: float,
    colors_info,
    skeleton_info,
    radius: int,
    thickness: int,
    draw_face_keypoints: bool,
    skip_exists: bool,
) -> None:
    if skip_exists and os.path.exists(out_path):
        return

    if len(colors_info) != keypoints.shape[0]:
        raise ValueError(
            f"Color info length {len(colors_info)} != keypoints length {keypoints.shape[0]}"
        )

    canvas = np.zeros((canvas_shape[1], canvas_shape[0], 3), dtype=np.uint8)

    # copy dict so we don't mutate the global constant
    sk_info = dict(skeleton_info)
    sk_info.update(
        {
            65: dict(link=(5, 12), id=65, color=BLUE),
            66: dict(link=(6, 11), id=66, color=BLUE),
        }
    )
    sk_items = list(sk_info.items())

    radii = np.ones(len(sk_items), dtype=np.int32) * int(radius)
    thicknesses = np.ones(len(sk_items), dtype=np.int32) * int(thickness)
    radii[:25] *= 2
    thicknesses[:25] *= 2

    lines: list[dict] = []
    w, h = canvas_shape
    for idx, (_skid, link_info) in enumerate(sk_items):
        i1, i2 = link_info["link"]
        p1 = keypoints[i1]
        p2 = keypoints[i2]
        if (p1 <= INVALID + 1e3).any() or (p2 <= INVALID + 1e3).any():
            continue
        
        p1_score = scores[i1]
        p2_score = scores[i2]
        line_score = min(p1_score, p2_score)
        if line_score < low_thr:
            continue

        # Clip line to image bounds
        clipped = _clip_line_to_bounds(p1, p2, w, h)
        if clipped is None:
            # Line doesn't intersect image bounds, skip
            continue
        
        clipped_p1, clipped_p2 = clipped
        
        line_color = score_to_color(link_info["color"], line_score, low_thr, high_thr)[::-1]
        p1_color = score_to_color(colors_info[i1], p1_score, low_thr, high_thr)[::-1]
        p2_color = score_to_color(colors_info[i2], p2_score, low_thr, high_thr)[::-1]
        depth = float((depths[i1] + depths[i2]) / 2.0)

        lines.append(
            {
                "p1": (int(round(clipped_p1[0])), int(round(clipped_p1[1]))),
                "p2": (int(round(clipped_p2[0])), int(round(clipped_p2[1]))),
                "p1_orig": p1,  # Original endpoint for circle drawing
                "p2_orig": p2,  # Original endpoint for circle drawing
                "line_color": line_color,
                "p1_color": p1_color,
                "p2_color": p2_color,
                "radius": int(radii[idx]),
                "thickness": int(thicknesses[idx]),
                "depth": depth,
                "score": line_score,
            }
        )

    if (depths > INVALID + 1e3).any():
        lines = sorted(lines, key=lambda x: x["depth"], reverse=True)
    else:
        lines = sorted(lines, key=lambda x: x["score"])

    for line in lines:
        cv2.line(canvas, line["p1"], line["p2"], line["line_color"], line["thickness"])
        # Only draw endpoint circles if they are within image bounds
        p1_orig = line["p1_orig"]
        p2_orig = line["p2_orig"]
        if 0 <= p1_orig[0] < w and 0 <= p1_orig[1] < h:
            cv2.circle(canvas, line["p1"], line["radius"], line["p1_color"], -1)
        if 0 <= p2_orig[0] < w and 0 <= p2_orig[1] < h:
            cv2.circle(canvas, line["p2"], line["radius"], line["p2_color"], -1)

    if draw_face_keypoints:
        w, h = canvas_shape
        for kid, (point, score) in enumerate(zip(keypoints, scores)):
            if not (23 < kid < 91):
                continue
            if score < low_thr or colors_info[kid] is None:
                continue
            # Check if point is within image bounds
            if point[0] < 0 or point[0] >= w or point[1] < 0 or point[1] >= h:
                continue
            color = score_to_color(colors_info[kid], score, low_thr, high_thr)[::-1]
            cv2.circle(canvas, (int(round(point[0])), int(round(point[1]))), radius, color, -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    success = cv2.imwrite(out_path, canvas)
    if not success:
        raise IOError(f"Failed to write {out_path}")


def _process_single_frame(
    pcd_path: str,
    tem_label: str,
    cameras: Sequence[dict],
    out_kpmap_dir: str,
    low_thr: float,
    high_thr: float,
    radius: int,
    thickness: int,
    draw_face_keypoints: bool,
    skip_exists: bool,
    colors_info,
    skeleton_info,
    image_ext: str,
    out_kp2d_dir: str | None,
):
    points = _load_ply_vertices(pcd_path)
    if points.shape[0] != len(colors_info):
        raise ValueError(
            f"Unexpected number of vertices ({points.shape[0]}) in {pcd_path}, "
            f"expected {len(colors_info)}."
        )
    for cam in cameras:
        canvas_shape = (cam["w"], cam["h"])
        kp2d, scores, depths = _project_points(points, cam)
        image_path = os.path.join(out_kpmap_dir, cam["label"], f"{tem_label}{image_ext}")
        _render_skeleton(
            kp2d,
            scores,
            depths,
            image_path,
            canvas_shape,
            low_thr,
            high_thr,
            colors_info,
            skeleton_info,
            radius,
            thickness,
            draw_face_keypoints,
            skip_exists,
        )
        if out_kp2d_dir is not None:
            os.makedirs(os.path.join(out_kp2d_dir, cam["label"]), exist_ok=True)
            json_path = os.path.join(out_kp2d_dir, cam["label"], f"{tem_label}.json")
            payload = {
                "instance_info": [
                    {
                        "keypoints": kp2d.tolist(),
                        "keypoint_scores": scores.tolist(),
                        "keypoint_depths": depths.tolist(),
                    }
                ]
            }
            with open(json_path, "w") as f:
                json.dump(payload, f)


def project_pcd_skeletons(
    pcd_dir: str,
    camera_path: str,
    out_kpmap_dir: str,
    spa_labels: Sequence[int] | None = None,
    tem_labels: Sequence[int] | None = None,
    image_ext: str = ".webp",
    num_workers: int = 8,
    low_thr: float = 0.5,
    high_thr: float = 0.9,
    radius: int = 2,
    thickness: int = 2,
    draw_face_keypoints: bool = False,
    skip_exists: bool = False,
    out_kp2d_dir: str | None = None,
    camera_coord_convention: str = "opengl",
):
    cameras = _load_cameras(camera_path, spa_labels, camera_coord_convention)

    if tem_labels is None:
        ply_paths = sorted(
            [
                os.path.join(pcd_dir, fname)
                for fname in os.listdir(pcd_dir)
                if fname.endswith(".ply")
            ]
        )
        tem_labels = [os.path.splitext(os.path.basename(p))[0] for p in ply_paths]
    else:
        tem_labels = [f"{tem:06d}" for tem in tem_labels]
        ply_paths = [os.path.join(pcd_dir, f"{tem_label}.ply") for tem_label in tem_labels]

    missing = [path for path in ply_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing PLY files: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    parallel_execution(
        ply_paths,
        tem_labels,
        action=_process_single_frame,
        num_workers=num_workers,
        print_progress=True,
        desc="Projecting 3D skeletons",
        sequential=False,
        cameras=tuple(cameras),
        out_kpmap_dir=out_kpmap_dir,
        low_thr=low_thr,
        high_thr=high_thr,
        radius=radius,
        thickness=thickness,
        draw_face_keypoints=draw_face_keypoints,
        skip_exists=skip_exists,
        colors_info=COCO_WHOLEBODY_KPTS_COLORS,
        skeleton_info=COCO_WHOLEBODY_SKELETON_INFO,
        image_ext=image_ext,
        out_kp2d_dir=out_kp2d_dir,
    )


if __name__ == "__main__":
    fire.Fire(project_pcd_skeletons)

