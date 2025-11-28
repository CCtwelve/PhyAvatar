#!/usr/bin/env python3
"""
shift_transforms_folder.py

将 Nerfstudio transforms.json 中 frames 的 file_path 中的文件夹编号整体加一。

示例：
    images_alpha/00/000000.png -> images_alpha/01/000000.png

使用方法：
    python shift_transforms_folder.py \
        --input /path/to/transforms.json \
        --output /path/to/transforms_shifted.json
"""

import argparse
import json
from pathlib import Path
from typing import Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 transforms.json 中 file_path 的子文件夹编号整体加 1。"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48/transforms.json"),
        help="原始 transforms.json 路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48/transforms_shifted.json"),
        help="输出路径；若不指定则覆盖输入文件",
    )
    parser.add_argument(
        "--prefix",
        default="images_alpha",
        help="需要处理的路径前缀，默认 images_alpha",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要修改的内容，不写回文件",
    )
    return parser.parse_args()


def shift_folder(path_str: str, prefix: str) -> Tuple[str, bool]:
    """
    将形如 prefix/XX/... 的路径中的文件夹编号 +1。

    Returns:
        (new_path, changed)
    """
    parts = path_str.split("/")
    if len(parts) < 2 or parts[0] != prefix:
        return path_str, False

    folder = parts[1]
    if not folder.isdigit():
        return path_str, False

    new_folder = f"{int(folder) + 1:02d}"
    parts[1] = new_folder
    return "/".join(parts), True


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or args.input

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    total = len(frames)
    modified = 0

    for frame in frames:
        file_path = frame.get("file_path")
        if not file_path:
            continue
        new_path, changed = shift_folder(file_path, args.prefix)
        if changed:
            frame["file_path"] = new_path
            modified += 1

    print(f"共处理 {total} 个 frame，修改 {modified} 条 file_path。")

    if args.dry_run:
        print("Dry run 模式，不写回文件。")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"已写入 {output_path}")


if __name__ == "__main__":
    main()

