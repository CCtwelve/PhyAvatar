import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Iterable, Sequence

import fire
import numpy as np
import open3d as o3d

FACE_RANGE = range(24, 91)  # COCO-WholeBody face keypoints (inclusive of 24, exclusive of 91)
INVALID = -1e5
FRAME_PADDING = 6


def _resolve_dir(base_dir: str, primary: str, fallback: str) -> str:
    """Return the first existing directory between primary and fallback names."""
    primary_path = os.path.join(base_dir, primary)
    if os.path.isdir(primary_path):
        return primary_path
    fallback_path = os.path.join(base_dir, fallback)
    if os.path.isdir(fallback_path):
        return fallback_path
    raise FileNotFoundError(
        f"Missing directory: tried '{primary}' and fallback '{fallback}' under {base_dir}"
    )


def _resolve_file(
    base_dir: str,
    primary_dir: str,
    fallback_dir: str,
    filename: str,
    label: str,
) -> str:
    """Return the first existing file between primary and fallback subdirectories."""
    primary_path = os.path.join(base_dir, primary_dir, filename)
    if os.path.isfile(primary_path):
        return primary_path
    fallback_path = os.path.join(base_dir, fallback_dir, filename)
    if os.path.isfile(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Missing {label}: {primary_path} or {fallback_path}")


@dataclass
class AlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    transform: np.ndarray


def _normalize_frame_id(frame: str | int, padding: int = FRAME_PADDING) -> str:
    """Ensure frame identifiers remain zero-padded strings for file lookups."""
    frame_str = str(frame)
    if frame_str.isdigit():
        return frame_str.zfill(padding)
    return frame_str


def _load_keypoints(json_path: str) -> np.ndarray:
    with open(json_path, "r") as f:
        data = json.load(f)
    instance = data["instance_info"][0]
    return np.asarray(instance["keypoints"], dtype=np.float64)


def _filter_indices(points: np.ndarray, indices: Iterable[int]) -> tuple[np.ndarray, list[int]]:
    valid_points = []
    kept_indices = []
    for idx in indices:
        point = points[idx]
        if np.all(point > INVALID):
            valid_points.append(point)
            kept_indices.append(idx)
    if not valid_points:
        raise ValueError("No valid keypoints available for alignment.")
    return np.asarray(valid_points, dtype=np.float64), kept_indices


def _compute_alignment_result(
    rotation: np.ndarray, src_centroid: np.ndarray, tgt_centroid: np.ndarray
) -> AlignmentResult:
    t = tgt_centroid - rotation @ src_centroid
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = t
    return AlignmentResult(rotation=rotation, translation=t, transform=T)


def _alignment_error(source: np.ndarray, target: np.ndarray, result: AlignmentResult) -> float:
    transformed = _apply_transform(source, result.rotation, result.translation)
    diff = transformed - target
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _compute_rigid_transform(source: np.ndarray, target: np.ndarray) -> AlignmentResult:
    if source.shape != target.shape:
        raise ValueError("Source and target keypoints must have identical shapes.")
    if source.shape[0] < 3:
        raise ValueError("Need at least 3 keypoints to compute a unique rigid transform.")

    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R_reflect = Vt.T @ U.T
    result_reflect = _compute_alignment_result(R_reflect, src_centroid, tgt_centroid)

    if np.linalg.det(R_reflect) >= 0:
        return result_reflect

    Vt_fixed = Vt.copy()
    Vt_fixed[-1, :] *= -1
    R_proper = Vt_fixed.T @ U.T
    result_proper = _compute_alignment_result(R_proper, src_centroid, tgt_centroid)

    err_reflect = _alignment_error(source, target, result_reflect)
    err_proper = _alignment_error(source, target, result_proper)
    return result_reflect if err_reflect <= err_proper else result_proper


def _apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points @ rotation.T) + translation


def _align_point_cloud(source_ply: str, output_ply: str, result: AlignmentResult) -> None:
    pcd = o3d.io.read_point_cloud(source_ply)
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in {source_ply}")

    aligned_points = _apply_transform(np.asarray(pcd.points), result.rotation, result.translation)
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)

    if pcd.has_colors():
        aligned_pcd.colors = pcd.colors
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        aligned_pcd.normals = o3d.utility.Vector3dVector(normals @ result.rotation.T)

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    o3d.io.write_point_cloud(output_ply, aligned_pcd)


def _align_pose_json(source_json: str, output_json: str, result: AlignmentResult) -> None:
    with open(source_json, "r") as f:
        data = json.load(f)
    keypoints = np.asarray(data["instance_info"][0]["keypoints"], dtype=np.float64)
    transformed = _apply_transform(keypoints, result.rotation, result.translation)
    data["instance_info"][0]["keypoints"] = transformed.tolist()
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)


def _align_all_point_clouds(source_dir: str, output_dir: str, result: AlignmentResult) -> list[str]:
    src_pcd_dir = _resolve_dir(source_dir, "poses_pcd_HQ", "poses_pcd")
    dst_pcd_dir = os.path.join(output_dir, "poses_pcd")
    os.makedirs(dst_pcd_dir, exist_ok=True)
    processed = []
    for fname in sorted(os.listdir(src_pcd_dir)):
        if not fname.endswith(".ply"):
            continue
        src_ply = os.path.join(src_pcd_dir, fname)
        dst_ply = os.path.join(dst_pcd_dir, fname)
        _align_point_cloud(src_ply, dst_ply, result)
        processed.append(fname)
    return processed


def _align_all_pose_jsons(source_dir: str, output_dir: str, result: AlignmentResult) -> list[str]:
    src_pose_dir = _resolve_dir(source_dir, "poses_3d_HQ", "poses_3d")
    dst_pose_dir = os.path.join(output_dir, "poses_3d")
    os.makedirs(dst_pose_dir, exist_ok=True)
    processed = []
    for fname in sorted(os.listdir(src_pose_dir)):
        if not fname.endswith(".json"):
            continue
        src_json = os.path.join(src_pose_dir, fname)
        dst_json = os.path.join(dst_pose_dir, fname)
        _align_pose_json(src_json, dst_json, result)
        processed.append(fname)
    return processed


def _align_transforms_json(source_json: str, output_json: str, result: AlignmentResult) -> None:
    with open(source_json, "r") as f:
        data = json.load(f)
    for frame in data.get("frames", []):
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        updated = result.transform @ matrix
        frame["transform_matrix"] = updated.tolist()
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)


def _fix_transforms_file_paths(transform_json: str) -> None:
    """修复 transforms.json 中的文件路径，确保使用 .webp 格式而不是 .jpg"""
    try:
        with open(transform_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fixed_count = 0
        for frame in data.get('frames', []):
            file_path = frame.get('file_path', '')
            if file_path.endswith('.jpg'):
                frame['file_path'] = file_path.replace('.jpg', '.webp')
                fixed_count += 1
            elif file_path.endswith('.jpeg'):
                frame['file_path'] = file_path.replace('.jpeg', '.webp')
                fixed_count += 1
            elif file_path.endswith('.png'):
                frame['file_path'] = file_path.replace('.png', '.webp')
                fixed_count += 1
        
        if fixed_count > 0:
            with open(transform_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        # 如果修复失败，不影响主流程
        pass


def _increment_camera_labels(transform_json: str) -> None:
    with open(transform_json, "r") as f:
        data = json.load(f)

    modified = False
    for frame in data.get("frames", []):
        label = frame.get("camera_label")
        if not label:
            continue
        match = re.search(r"(\d+)$", label)
        if not match:
            continue
        prefix = label[: match.start(1)]
        digits = match.group(1)
        incremented = str(int(digits)).zfill(len(digits))
        new_label = f"{prefix}{incremented}"
        if new_label != label:
            frame["camera_label"] = new_label
            modified = True

    if modified:
        with open(transform_json, "w") as f:
            json.dump(data, f, indent=2)


def _refine_with_icp(
    source_ply: str,
    target_ply: str,
    result: AlignmentResult,
    max_distance: float,
    max_iterations: int,
    voxel_size: float | None,
) -> AlignmentResult:
    src_pcd = o3d.io.read_point_cloud(source_ply)
    tgt_pcd = o3d.io.read_point_cloud(target_ply)
    if len(src_pcd.points) == 0 or len(tgt_pcd.points) == 0:
        raise ValueError("ICP refinement requires non-empty source and target point clouds.")

    src_pcd.transform(result.transform)
    if voxel_size:
        src_down = src_pcd.voxel_down_sample(voxel_size)
        tgt_down = tgt_pcd.voxel_down_sample(voxel_size)
    else:
        src_down = src_pcd
        tgt_down = tgt_pcd

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    icp = o3d.pipelines.registration.registration_icp(
        src_down,
        tgt_down,
        max_distance,
        np.eye(4),
        estimation,
        criteria,
    )
    refined_T = icp.transformation @ result.transform
    refined_R = refined_T[:3, :3]
    refined_t = refined_T[:3, 3]
    return AlignmentResult(rotation=refined_R, translation=refined_t, transform=refined_T)


def align_pose_pcd(
    source_dir: str,
    target_dir: str,
    source_frame: str | int = "000000",
    target_frame: str | int = "000000",
    transform_name: str = "transforms_HQ.json",
    output_dir: str | None = None,
    face_indices: Sequence[int] = FACE_RANGE,
    run_icp: bool = False,
    icp_max_distance: float = 0.01,
    icp_max_iterations: int = 50,
    icp_voxel_size: float | None = 0.005,
) -> dict:
    """
    Align a source pose to a target pose using face keypoints and apply the transform
    to the source point cloud, 3D keypoints, and camera transforms.
    """
    source_frame = _normalize_frame_id(source_frame)
    target_frame = _normalize_frame_id(target_frame)

    source_pose_json = _resolve_file(
        source_dir,
        "poses_3d_HQ",
        "poses_3d",
        f"{source_frame}.json",
        "source pose json",
    )
    target_pose_json = os.path.join(target_dir, "poses_3d", f"{target_frame}.json")
    source_ply = _resolve_file(
        source_dir,
        "poses_pcd_HQ",
        "poses_pcd",
        f"{source_frame}.ply",
        "source point cloud",
    )
    target_ply = os.path.join(target_dir, "poses_pcd", f"{target_frame}.ply")
    source_transform_json = os.path.join(source_dir, transform_name)
    target_transform_json = os.path.join(target_dir, "transforms.json")

    if output_dir is None:
        target_tag = os.path.basename(os.path.normpath(target_dir))
        output_dir = os.path.join(source_dir)

    os.makedirs(output_dir, exist_ok=True)
    target_transform_copy = os.path.join(output_dir, "transforms.json")
    if os.path.isfile(target_transform_json):
        shutil.copyfile(target_transform_json, target_transform_copy)
        # 修复文件路径格式，确保使用 .webp 而不是 .jpg
        _fix_transforms_file_paths(target_transform_copy)
        # _increment_camera_labels(target_transform_copy)
    else:
        raise FileNotFoundError(f"Missing target transforms: {target_transform_json}")

    source_keypoints = _load_keypoints(source_pose_json)
    target_keypoints = _load_keypoints(target_pose_json)

    src_face, kept_indices = _filter_indices(source_keypoints, face_indices)
    tgt_face, _ = _filter_indices(target_keypoints, kept_indices)

    result = _compute_rigid_transform(src_face, tgt_face)
    if run_icp:
        result = _refine_with_icp(
            source_ply,
            target_ply,
            result,
            max_distance=icp_max_distance,
            max_iterations=icp_max_iterations,
            voxel_size=icp_voxel_size,
        )

    output_transform_json = os.path.join(output_dir, "transforms_HQ2DNARender.json")

    aligned_pcd_frames = _align_all_point_clouds(source_dir, output_dir, result)
    aligned_pose_frames = _align_all_pose_jsons(source_dir, output_dir, result)
    _align_transforms_json(source_transform_json, output_transform_json, result)

    return {
        "rotation": result.rotation.tolist(),
        "translation": result.translation.tolist(),
        "transform": result.transform.tolist(),
        "output_dir": output_dir,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "aligned_pcd_frames": aligned_pcd_frames,
        "aligned_pose_frames": aligned_pose_frames,
        "target_transform_copy": target_transform_copy,
    }


if __name__ == "__main__":
    fire.Fire(align_pose_pcd)

"""
python scripts/preprocess_/align_pose_pcd.py \
  --source_dir /mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_48 \
  --target_dir /mnt/cvda/cvda_phava/code/Han/Diffuman4D/data/datasets--krahets--diffuman4d_example/0023_06 \
  --source_frame "000000" \
  --target_frame "000000" \
  --transform_name transforms_48_z.json \
  --output_dir /mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_5_2_48/nerfstudio_5_2_48_z_aligned \
  --run_icp True \
  --icp_max_distance 0.02 \
  --icp_voxel_size 0.005
"""