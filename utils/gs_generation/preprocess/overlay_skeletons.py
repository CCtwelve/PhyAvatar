from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import cv2
import fire
import numpy as np


def _normalize_extensions(exts: Sequence[str]) -> Tuple[str, ...]:
    return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts)


def _collect_files(root: str, extensions: Sequence[str]) -> List[Tuple[str, str]]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    extensions = _normalize_extensions(extensions)
    results: List[Tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in extensions:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root)
                results.append((rel_path, full_path))
    return sorted(results)


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        # Drop alpha for overlay compositing.
        bgr = image[:, :, :3]
        alpha = image[:, :, 3:] / 255.0
        background = np.zeros_like(bgr)
        image = (alpha * bgr + (1 - alpha) * background).astype(np.uint8)
    return image


def _find_matching_image(images_dir: str, rel_path: str, candidate_exts: Sequence[str]) -> str | None:
    base_rel = os.path.splitext(rel_path)[0]
    candidates = [rel_path]
    for ext in candidate_exts:
        ext = ext if ext.startswith(".") else f".{ext}"
        candidates.append(f"{base_rel}{ext.lower()}")
        candidates.append(f"{base_rel}{ext.upper()}")
    for candidate in candidates:
        image_path = os.path.join(images_dir, candidate)
        if os.path.exists(image_path):
            return image_path
    return None


def _overlay_single(image_path: str, skeleton_path: str, alpha: float, beta: float) -> np.ndarray:
    image = _read_image(image_path)
    skeleton = _read_image(skeleton_path)
    if image.shape[:2] != skeleton.shape[:2]:
        skeleton = cv2.resize(skeleton, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(image, alpha, skeleton, beta, 0)


def _overlay_directory(
    label: str,
    images_dir: str,
    skeleton_dir: str,
    out_dir: str,
    alpha: float,
    beta: float,
    skeleton_exts: Sequence[str],
    image_search_exts: Sequence[str],
    skip_missing: bool,
) -> List[str]:
    if not os.path.isdir(skeleton_dir):
        raise FileNotFoundError(f"[{label}] Skeleton directory not found: {skeleton_dir}")
    os.makedirs(out_dir, exist_ok=True)

    skeleton_files = _collect_files(skeleton_dir, skeleton_exts)
    produced_paths: List[str] = []
    for rel_path, skeleton_path in skeleton_files:
        image_path = _find_matching_image(images_dir, rel_path, image_search_exts)
        if image_path is None:
            message = f"[{label}] Missing matching image for {rel_path}"
            if skip_missing:
                print(f">> Warning: {message}, skipping.")
                continue
            raise FileNotFoundError(message)

        overlay = _overlay_single(image_path, skeleton_path, alpha, beta)
        out_path = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        success = cv2.imwrite(out_path, overlay)
        if not success:
            raise IOError(f"[{label}] Failed to write overlay: {out_path}")
        produced_paths.append(out_path)
    print(f">> [{label}] Generated {len(produced_paths)} overlay images.")
    return produced_paths


def _pad_to_dimension(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    top = max((target_h - h) // 2, 0)
    bottom = max(target_h - h - top, 0)
    left = max((target_w - w) // 2, 0)
    right = max(target_w - w - left, 0)
    if top == bottom == left == right == 0:
        return image
    return cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _stack_row(image_paths: Sequence[str]) -> np.ndarray | None:
    if not image_paths:
        return None
    images = [_read_image(path) for path in image_paths]
    max_h = max(img.shape[0] for img in images)
    padded = [_pad_to_dimension(img, max_h, img.shape[1]) for img in images]
    return np.concatenate(padded, axis=1)


def _build_collage(
    top_paths: Sequence[str],
    bottom_paths: Sequence[str],
    collage_path: str,
):
    top_row = _stack_row(top_paths)
    bottom_row = _stack_row(bottom_paths)
    if top_row is None and bottom_row is None:
        print(">> No overlays available; skipping collage generation.")
        return
    if top_row is None:
        collage = bottom_row
    elif bottom_row is None:
        collage = top_row
    else:
        target_width = max(top_row.shape[1], bottom_row.shape[1])
        top_row = _pad_to_dimension(top_row, top_row.shape[0], target_width)
        bottom_row = _pad_to_dimension(bottom_row, bottom_row.shape[0], target_width)
        collage = np.concatenate([top_row, bottom_row], axis=0)
    os.makedirs(os.path.dirname(collage_path), exist_ok=True)
    success = cv2.imwrite(collage_path, collage)
    if not success:
        raise IOError(f"Failed to write collage: {collage_path}")
    print(f">> Collage saved to {collage_path}")


def overlay_skeletons(
    datadir: str,
    images_dir: str | None = None,
    project_skeleton_dir: str | None = None,
    poses_skeleton_dir: str | None = None,
    project_overlay_dir: str | None = None,
    poses_overlay_dir: str | None = None,
    collage_output_path: str | None = None,
    skeleton_extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
    image_search_extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
    alpha: float = 0.65,
    beta: float = 0.35,
    skip_missing: bool = True,
):
    datadir = os.path.abspath(datadir)
    images_dir = images_dir or os.path.join(datadir, "images")
    project_skeleton_dir = project_skeleton_dir or os.path.join(datadir, "project_pcd_skeletons")
    poses_skeleton_dir = poses_skeleton_dir or os.path.join(datadir, "skeletons")
    project_overlay_dir = project_overlay_dir or os.path.join(datadir, "overlay_project_pcd")
    poses_overlay_dir = poses_overlay_dir or os.path.join(datadir, "overlay_poses_sapiens")
    collage_output_path = collage_output_path or os.path.join(datadir, "skeleton_overlay_collage.png")

    print(">> Overlay configuration:")
    print(f"   Images dir: {images_dir}")
    print(f"   Project skeleton dir: {project_skeleton_dir}")
    print(f"   Poses skeleton dir: {poses_skeleton_dir}")
    project_overlays = _overlay_directory(
        label="project_pcd",
        images_dir=images_dir,
        skeleton_dir=project_skeleton_dir,
        out_dir=project_overlay_dir,
        alpha=alpha,
        beta=beta,
        skeleton_exts=skeleton_extensions,
        image_search_exts=image_search_extensions,
        skip_missing=skip_missing,
    )
    poses_overlays = _overlay_directory(
        label="poses_sapiens",
        images_dir=images_dir,
        skeleton_dir=poses_skeleton_dir,
        out_dir=poses_overlay_dir,
        alpha=alpha,
        beta=beta,
        skeleton_exts=skeleton_extensions,
        image_search_exts=image_search_extensions,
        skip_missing=skip_missing,
    )
    _build_collage(project_overlays, poses_overlays, collage_output_path)


if __name__ == "__main__":
    fire.Fire(overlay_skeletons)


