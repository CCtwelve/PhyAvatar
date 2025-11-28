import os
import shutil
import logging
import json
import re
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import yaml

DEFAULT_COLMAP2NERF_ARGS = dict(
    root=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/colmap"),
    target=Path("/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_4_2_48"),
    is_ba="0",
)

def init_parser():
    parser = ArgumentParser("Nerfstudio 转换工具：将 BA 后的 COLMAP 模型转为 Nerfstudio 格式")

    parser.add_argument(
        "--config",
        default=Path("/mnt/cvda/cvda_phava/code/Han/PhyAvatar/config/Diffuman/Hq_4_2_DNA_48.py"),
        type=Path,
        help="包含 colmap2nerf 参数的配置文件（支持 .py 或 .yaml）。",
    )
    parser.add_argument("--root", type=Path, help="COLMAP 项目根目录（优先于 YAML 配置）")
    parser.add_argument("--target", type=Path, help="最终输出目录（优先于 YAML 配置）")
    parser.add_argument("--is_ba", help="sparse 子文件夹名（优先于 YAML 配置）")

    return parser


def load_colmap2nerf_args(config_path: Path):
    """
    从 Python 或 YAML 配置文件中加载 colmap2nerf 的默认参数。
    """
    config = DEFAULT_COLMAP2NERF_ARGS.copy()

    if not config_path.exists():
        print(f"警告: 配置文件 {config_path} 不存在，使用默认参数。")
        return config

    # Load from Python or YAML file
    if config_path.suffix == '.py':
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            raw_cfg = getattr(config_module, 'config', {})
        except Exception as exc:
            print(f"警告: 加载 Python 配置失败（{exc}），使用默认参数。")
            return config
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"警告: 加载 YAML 配置失败（{exc}），使用默认参数。")
            return config

    cfg_section = raw_cfg.get("colmap2nerf_args")
    if isinstance(cfg_section, dict):
        config.update(cfg_section)

    print(f"已从 {config_path} 加载 colmap2nerf 参数。")
    return config

def copy_and_reorganize_images(source_images_dir: Path, target_base_dir: Path, target_size=(1024, 1024)):
    """
    读取源图片，调整尺寸(居中填充黑边)，转为无损webp，并按结构存储。
    """
    target_images_dir = target_base_dir / 'images'
    target_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- 1. 图片处理 ---")
    print(f"源目录: {source_images_dir}")
    print(f"目标: {target_images_dir}")
    
    # 匹配 frame_000xx.jpg 格式
    pattern = re.compile(r'frame_(\d+)\.jpg$', re.IGNORECASE)
    
    if not source_images_dir.exists():
        print(f"\n\033[91m错误: 源图片目录不存在: {source_images_dir}\033[0m")
        return {}
    
    image_files = []
    for file_path in source_images_dir.iterdir():
        if file_path.is_file():
            match = pattern.search(file_path.name)
            if match:
                frame_num = int(match.group(1)) - 1
                image_files.append((file_path, frame_num))
        
    image_files.sort(key=lambda x: x[1])
    
    if not image_files:
        print(f"\n\033[93m警告: 未找到匹配 frame_xxxxx.jpg 的图片\033[0m")
        return {}
    
    print(f"找到 {len(image_files)} 张图片")
    
    original_sizes = {}
    target_w, target_h = target_size
    
    for source_file, frame_num in image_files:
        camera_label = f"{frame_num:02d}"
        target_subdir = target_images_dir / camera_label
        target_subdir.mkdir(parents=True, exist_ok=True)
        target_file = target_subdir / "000000.webp"
        
        try:
            img = Image.open(source_file)
            original_w, original_h = img.size
            original_sizes[camera_label] = (original_w, original_h)
            
            # 统一转为 RGBA 或 RGB
            if img.mode != 'RGBA' and img.mode != 'RGB':
                img = img.convert('RGBA')
            
            # 创建黑色背景
            mode = 'RGBA' if img.mode == 'RGBA' else 'RGB'
            bg_color = (0, 0, 0, 255) if mode == 'RGBA' else (0, 0, 0)
            new_img = Image.new(mode, (target_w, target_h), bg_color)

            # 居中粘贴
            offset_x = (target_w - original_w) // 2
            offset_y = (target_h - original_h) // 2
            new_img.paste(img, (offset_x, offset_y))

            new_img.save(target_file, 'WEBP', lossless=True, method=6)
            
        except Exception as e:
            print(f"  错误: 处理 {source_file.name} 失败: {e}")
    
    print(f"\033[92m图片处理完成。\033[0m")
    return original_sizes

def process_and_copy_transforms_json(source_json_path: Path, target_json_path: Path, original_sizes: dict = None, target_size=(1024, 1024)):
    """
    修改 transforms.json：更新路径、尺寸和内参 (cx, cy)。
    """
    if not source_json_path.exists():
        print(f"\n\033[93m警告: transforms.json 不存在\033[0m")
        return
    
    print(f"\n--- 2. 处理 transforms.json ---")
    if original_sizes is None: original_sizes = {}
    target_w, target_h = target_size
    
    try:
        with open(source_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['ply_file_path'] = "sparse_pcd.ply"
        
        valid_frames = []
        for frame in data.get('frames', []):
            file_path = frame.get('file_path', '')
            # 从路径中提取编号
            match = re.search(r'frame_(\d+)', file_path)
            if not match:
                continue
            
            frame_num = int(match.group(1)) - 1
            camera_label = f"{frame_num:02d}"
            
            frame['camera_label'] = camera_label
            frame['file_path'] = f"images/{camera_label}/000000.webp"
            
            # 更新内参
            if camera_label in original_sizes:
                orig_w, orig_h = original_sizes[camera_label]
                off_x = (target_w - orig_w) / 2.0
                off_y = (target_h - orig_h) / 2.0
                
                frame['w'] = target_w
                frame['h'] = target_h
                frame['cx'] = float(frame['cx']) + off_x
                frame['cy'] = float(frame['cy']) + off_y
            else:
                frame['w'] = target_w
                frame['h'] = target_h
            
            valid_frames.append(frame)
            
        data['frames'] = valid_frames
        
        target_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"\033[92mtransforms.json 处理完成，保存至 {target_json_path}\033[0m")
        
    except Exception as e:
        print(f"\033[91mJSON 处理出错: {e}\033[0m")

def main():
    parser = init_parser()
    args = parser.parse_args()
    cfg_args = load_colmap2nerf_args(Path(args.config))

    _root = Path(args.root) if args.root else Path(cfg_args["root"])
    _target = Path(args.target) if args.target else Path(cfg_args["target"])
    is_ba = args.is_ba if args.is_ba is not None else cfg_args["is_ba"]
    # 与 colmap_construction 保持一致，使用倒置后的图片目录
    _images_path = _root / 'images'
    _ba_model_path = _root / 'sparse' / f"{is_ba}"
    
    # 临时输出目录 (ns-process-data 的直接输出)
    _nerfstudio_temp = _root / 'nerfstudio'

    print(f"--- COLMAP 转 Nerfstudio ---")
    
    if not _images_path.exists():
        print(f"错误: 图片路径不存在 {_images_path}")
        return

    # 1. 运行 ns-process-data
    if _nerfstudio_temp.exists():
        try:
            shutil.rmtree(_nerfstudio_temp)
        except:
            pass
    _nerfstudio_temp.mkdir(parents=True, exist_ok=True)

    try:
        _ba_rel = os.path.relpath(_ba_model_path, _nerfstudio_temp)
    except:
        _ba_rel = _ba_model_path

    # 注意：ns-process-data 会自动对齐中心和缩放
    cmd = (
        f"ns-process-data images "
        f"--data {_images_path} "
        f"--output-dir {_nerfstudio_temp} "
        f"--colmap-model-path {_ba_rel} "
        f"--skip-colmap "
        f"--verbose"
    )
    
    print(f"执行命令: {cmd}")
    if os.system(cmd) != 0:
        print("ns-process-data 执行失败")
        return

    # 2. 图片后处理 (Resizing & WebP)
    orig_sizes = copy_and_reorganize_images(_nerfstudio_temp / 'images', _target)
    
    # 3. 处理 Transforms JSON
    source_json = _nerfstudio_temp / 'transforms.json'
    target_json = _target / 'transforms_HQ.json'
    process_and_copy_transforms_json(source_json, target_json, orig_sizes)

    print(f"\n\033[92mnerfstudio 转换完成。输出目录: {_target}\033[0m")


    
if __name__ == "__main__":
    main()