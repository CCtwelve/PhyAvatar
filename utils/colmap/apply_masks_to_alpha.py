#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use mask images to cut backgrounds and save alpha images with matching folder structure."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_4_2_39_HQ/images"),
        help="Root directory containing RGB images.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_4_2_39_HQ/fmasks"),
        help="Root directory containing binary masks (same folder layout as images).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_4_2_39_HQ/images_alpha"),
        help="Destination root to save RGBA images.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=128,
        help="Mask pixels > threshold are treated as foreground (alpha=255).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            yield path


def apply_mask(image_path: Path, mask_path: Path, output_path: Path, threshold: int, overwrite: bool) -> bool:
    if not mask_path.exists():
        print(f"[WARN] Missing mask for {image_path} -> {mask_path}", file=sys.stderr)
        return False
    if output_path.exists() and not overwrite:
        return False

    try:
        with Image.open(image_path) as img, Image.open(mask_path) as mask:
            rgba = img.convert("RGBA")
            mask_arr = np.array(mask.convert("L"))
            alpha = np.where(mask_arr > threshold, 255, 0).astype(np.uint8)

            rgba_arr = np.array(rgba)
            if rgba_arr.shape[:2] != alpha.shape:
                print(
                    f"[WARN] Shape mismatch {image_path} {rgba_arr.shape[:2]} vs {mask_path} {alpha.shape}",
                    file=sys.stderr,
                )
                return False

            rgba_arr[..., 3] = alpha
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba_arr).save(output_path)
            return True
    except Exception as err:  # pragma: no cover - quick utility script
        print(f"[ERROR] Failed to process {image_path}: {err}", file=sys.stderr)
        return False


def main():
    args = parse_args()

    if not args.image_root.exists():
        print(f"Image root does not exist: {args.image_root}", file=sys.stderr)
        return 1
    if not args.mask_root.exists():
        print(f"Mask root does not exist: {args.mask_root}", file=sys.stderr)
        return 1

    images = list(iter_image_files(args.image_root))
    if not images:
        print(f"No images found in {args.image_root}")
        return 0

    converted = 0
    for img_path in tqdm(images, desc="Applying masks"):
        rel_path = img_path.relative_to(args.image_root)
        mask_path = args.mask_root / rel_path
        output_path = args.output_root / rel_path.with_suffix(".png")
        if apply_mask(img_path, mask_path, output_path, args.mask_threshold, args.force):
            converted += 1

    print(f"Processed {len(images)} images, wrote {converted} RGBA files to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

