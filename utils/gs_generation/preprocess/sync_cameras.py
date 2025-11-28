import argparse
import json
import pathlib
import shutil
import sys
from pathlib import PurePosixPath
from typing import Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize camera transforms and duplicate image/mask folders."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=pathlib.Path,
        help="Root dataset directory (must contain images/, fmasks/, transforms files, etc.).",
    )
    parser.add_argument(
        "--source_transforms",
        default="transforms_HQ2DNARender.json",
        type=str,
        help="Transforms JSON (relative to data_dir) providing the source camera params.",
    )
    parser.add_argument(
        "--target_transforms",
        default="transforms.json",
        type=str,
        help="Transforms JSON (relative to data_dir) to be updated with additional cameras.",
    )
    parser.add_argument(
        "--source_labels",
        nargs="+",
        default=["04", "05", "09", "17"],
        help="Camera labels to copy from source transforms / directories.",
    )
    parser.add_argument(
        "--target_labels",
        nargs="+",
        default=["49", "50", "51", "52"],
        help="Camera labels to write into the target transforms / directories.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["images", "fmasks", "skeleton_gt:skeletons"],
        help=(
            "Dataset subdirectories containing per-camera folders to duplicate. "
            "Use src:dst syntax (e.g. skeleton_gt:skeletons) to copy between "
            "different parent directories."
        ),
    )
    parser.add_argument(
        "--folder_offset",
        type=int,
        default=0,
        help="Offset applied to camera labels when resolving folder names (e.g. 0 -> 01).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target cameras without prompting.",
    )
    return parser.parse_args()


def with_offset(label: str, offset: int) -> str:
    if offset == 0:
        return label
    try:
        width = len(label)
        value = int(label) + offset
        return f"{value:0{width}d}"
    except ValueError:
        return label


def detect_media_extension(folder: pathlib.Path) -> Optional[str]:
    if not folder.is_dir():
        return None
    for child in sorted(folder.iterdir()):
        if child.is_file():
            return child.suffix or None
    return None


def rewrite_media_path(
    path_str: Optional[str], root_dir: str, new_folder: str, new_ext: Optional[str]
) -> Optional[str]:
    if not path_str:
        return path_str

    posix_path = PurePosixPath(path_str)
    parts = list(posix_path.parts)
    if len(parts) < 3 or parts[0] != root_dir:
        return path_str

    parts[1] = new_folder
    filename = parts[-1]
    name_path = PurePosixPath(filename)
    ext = new_ext
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    if ext:
        parts[-1] = f"{name_path.stem}{ext}"
    return str(PurePosixPath(*parts))


def build_mapping(
    src_labels: Sequence[str], dst_labels: Sequence[str]
) -> List[Tuple[str, str]]:
    if len(src_labels) != len(dst_labels):
        raise ValueError("source_labels and target_labels must have the same length")
    return list(zip(src_labels, dst_labels))


def load_frames_by_label(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text())
    frames = data.get("frames", [])
    by_label = {}
    for frame in frames:
        label = frame.get("camera_label")
        if not label:
            continue
        by_label.setdefault(label, []).append(frame)
    return {"data": data, "by_label": by_label}


def should_overwrite(dst_frames: List[dict], target_labels: set[str], force: bool) -> bool:
    existing = {
        frame.get("camera_label") for frame in dst_frames if frame.get("camera_label")
    }
    duplicates = sorted(existing & target_labels)
    if not duplicates:
        return True
    if force:
        print(
            f"[sync_cameras] Force replacing existing camera labels: {', '.join(duplicates)}"
        )
        return True

    prompt = (
        "[sync_cameras] The following camera labels already exist in target "
        f"transforms: {', '.join(duplicates)}. Replace them? [y/N]: "
    )
    response = input(prompt).strip().lower()
    return response in {"y", "yes"}


def sync_transforms(
    data_dir: pathlib.Path,
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    mapping: Iterable[Tuple[str, str]],
    folder_offset: int,
    force: bool,
) -> int:
    src_payload = load_frames_by_label(src_path)
    dst_payload = load_frames_by_label(dst_path)

    src_by_label = src_payload["by_label"]
    dst_data = dst_payload["data"]
    target_labels = {dst for _, dst in mapping}
    dst_frames = dst_data.get("frames", [])

    if not should_overwrite(dst_frames, target_labels, force):
        print("[sync_cameras] Skip camera sync: user declined overwrite.")
        return 0

    dst_frames = [
        frame for frame in dst_frames if frame.get("camera_label") not in target_labels
    ]

    added = 0
    for src_label, dst_label in mapping:
        frames = src_by_label.get(src_label)
        if not frames:
            print(
                f"[sync_cameras] Warning: missing camera_label {src_label} in source transforms",
                file=sys.stderr,
            )
            continue
        frame = json.loads(json.dumps(frames[0]))  # deep copy
        frame["camera_label"] = dst_label

        src_folder = with_offset(src_label, folder_offset)
        dst_folder = with_offset(dst_label, folder_offset)

        image_ext = detect_media_extension(data_dir / "images" / src_folder)
        mask_ext = detect_media_extension(data_dir / "fmasks" / src_folder)

        frame["file_path"] = rewrite_media_path(
            frame.get("file_path"), "images", dst_folder, image_ext
        )
        if "mask_path" in frame:
            frame["mask_path"] = rewrite_media_path(
                frame.get("mask_path"), "fmasks", dst_folder, mask_ext
            )

        dst_frames.append(frame)
        added += 1

    dst_data["frames"] = dst_frames
    dst_path.write_text(json.dumps(dst_data, indent=2) + "\n")
    return added


def parse_subdir_spec(spec: str) -> Tuple[str, str]:
    if ":" in spec:
        src, dst = spec.split(":", 1)
        return src.strip(), dst.strip()
    return spec, spec


def copy_camera_dirs(
    data_dir: pathlib.Path,
    subdirs: Sequence[str],
    mapping: Iterable[Tuple[str, str]],
    folder_offset: int,
) -> None:
    for spec in subdirs:
        src_subdir, dst_subdir = parse_subdir_spec(spec)
        for src_label, dst_label in mapping:
            src_folder = with_offset(src_label, folder_offset)
            dst_folder = with_offset(dst_label, folder_offset)
            src_path = data_dir / src_subdir / src_folder
            dst_path = data_dir / dst_subdir / dst_folder
            if not src_path.is_dir():
                print(f"[sync_cameras] Skip copying: {src_path} not found")
                continue
            if dst_path.exists():
                print(f"[sync_cameras] Skip copying: {dst_path} already exists")
                continue
            print(f"[sync_cameras] Copying {src_path} -> {dst_path}")
            shutil.copytree(src_path, dst_path)


def main() -> int:
    args = parse_args()
    mapping = build_mapping(args.source_labels, args.target_labels)

    src_path = args.data_dir / args.source_transforms
    dst_path = args.data_dir / args.target_transforms

    if not src_path.exists() or not dst_path.exists():
        print(
            "[sync_cameras] Skip: transforms files missing "
            f"(source={src_path}, target={dst_path})"
        )
        return 0

    added = sync_transforms(
        args.data_dir,
        src_path,
        dst_path,
        mapping,
        args.folder_offset,
        args.force,
    )
    print(
        f"[sync_cameras] Added {added} camera entries into {args.target_transforms}"
    )

    if added:
        copy_camera_dirs(args.data_dir, args.subdirs, mapping, args.folder_offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

