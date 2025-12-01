import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import fire
from PIL import Image


def _parse_color(color: str) -> tuple[int, int, int]:
    # Fire 可能会把参数解析成 int/其他类型，这里统一转成字符串
    color = str(color).strip()
    if not color:
        color = "#000000"

    # 允许不带 # 的简写，比如 "000" 或 "000000"
    if not color.startswith("#"):
        color = "#" + color

    color = color[1:]
    if len(color) not in (3, 6):
        # 长度不对时直接回退到黑色，避免抛异常
        color = "000000"

    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)

    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return r, g, b


def _gather_images(
    images_dir: Path, extensions: Sequence[str] | None, recursive: bool
) -> list[Path]:
    if extensions is not None:
        extensions = tuple(ext.lower() for ext in extensions)
    iterator: Iterable[Path]
    iterator = images_dir.rglob("*") if recursive else images_dir.glob("*")
    image_paths = [
        path
        for path in iterator
        if path.is_file() and (extensions is None or path.suffix.lower() in extensions)
    ]
    return sorted(image_paths)


def _normalize_to_sequence(value):
    """Ensure CLI values (which Fire may coerce into scalars) behave like sequences."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        iter(value)  # type: ignore[arg-type]
    except TypeError:
        return [value]
    # Fire might coerce lists into tuples; ensure we can append later
    return list(value)  # type: ignore[arg-type]


def build_skeleton_collage(
    images_dir: str,
    output_path: str,
    images_per_row: int = 8,
    padding: int = 8,
    background_color: str = "#000000",
    recursive: bool = True,
    extensions: Sequence[str] | None = (".png", ".jpg", ".jpeg", ".webp"),
    comparison_images_dir: str | None = None,
    exclude_labels: Sequence[str] | None = None,
) -> str:
    """
    Combine skeleton renderings into a single collage image.
    """
    if images_per_row <= 0:
        raise ValueError("images_per_row must be > 0")

    def _open_images(path_str: str) -> list[tuple[Path, Image.Image]]:
        src_dir = Path(path_str)
        if not src_dir.exists():
            raise FileNotFoundError(f"{path_str} does not exist")
        collected = _gather_images(src_dir, extensions, recursive)
        if not collected:
            raise FileNotFoundError(f"No images found under {path_str}")
        opened: list[tuple[Path, Image.Image]] = []
        for img_path in collected:
            rel_path = img_path.relative_to(src_dir)
            opened.append((rel_path, Image.open(img_path).convert("RGB")))
        return opened

    primary_named_images = _open_images(images_dir)
    
    # Filter out excluded labels
    normalized_excludes = _normalize_to_sequence(exclude_labels)
    if normalized_excludes:
        # Create a set of all possible formats for excluded labels
        exclude_set = set()
        for label in normalized_excludes:
            label_str = str(label)
            exclude_set.add(label_str)
            # Also add zero-padded format (e.g., "47" -> "47", "7" -> "07")
            try:
                label_num = int(label_str)
                exclude_set.add(f"{label_num:02d}")
            except ValueError:
                pass
        primary_named_images = [
            (rel_path, img)
            for rel_path, img in primary_named_images
            if len(rel_path.parts) == 0 or rel_path.parts[0] not in exclude_set
        ]
    
    primary_images: list[Image.Image] = []
    comparison_images: list[Image.Image] = []
    if comparison_images_dir:
        comp_dir = Path(comparison_images_dir)
        if not comp_dir.exists():
            raise FileNotFoundError(f"{comparison_images_dir} does not exist")

        missing: list[str] = []
        for rel_path, img in primary_named_images:
            # Use the same path (no increment) for comparison
            comparison_path = comp_dir / rel_path
            if not comparison_path.is_file():
                missing.append(str(rel_path))
                continue
            primary_images.append(img)
            comparison_images.append(Image.open(comparison_path).convert("RGB"))

        if missing and not primary_images:
            missing_preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                "Comparison directory is missing all requested images. "
                f"First missing entries: {missing_preview}"
            )
        if missing:
            print(
                f"[collage_skeletons] Skipped {missing} files not found in comparison set"
            )
    else:
        primary_images = [img for _, img in primary_named_images]

    if not primary_images:
        raise FileNotFoundError(
            "No overlapping image names between primary and comparison directories"
        )

    opened_images: list[Image.Image] = []
    if comparison_images:
        primary_idx = 0
        comparison_idx = 0
        while primary_idx < len(primary_images) or comparison_idx < len(
            comparison_images
        ):
            if primary_idx < len(primary_images):
                opened_images.extend(
                    primary_images[
                        primary_idx : primary_idx + images_per_row
                    ]
                )
                primary_idx += images_per_row
            if comparison_idx < len(comparison_images):
                opened_images.extend(
                    comparison_images[
                        comparison_idx : comparison_idx + images_per_row
                    ]
                )
                comparison_idx += images_per_row
    else:
        opened_images = primary_images

    sizes: list[tuple[int, int]] = [img.size for img in opened_images]

    max_w = max(size[0] for size in sizes)
    max_h = max(size[1] for size in sizes)

    rows = math.ceil(len(opened_images) / images_per_row)
    canvas_w = images_per_row * max_w + padding * (images_per_row + 1)
    canvas_h = rows * max_h + padding * (rows + 1)
    bg_color = _parse_color(background_color)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    for idx, img in enumerate(opened_images):
        row = idx // images_per_row
        col = idx % images_per_row
        base_x = padding + col * (max_w + padding)
        base_y = padding + row * (max_h + padding)
        w, h = img.size
        offset_x = base_x + (max_w - w) // 2
        offset_y = base_y + (max_h - h) // 2
        canvas.paste(img, (offset_x, offset_y))

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    return output_path


if __name__ == "__main__":
    fire.Fire(build_skeleton_collage)

