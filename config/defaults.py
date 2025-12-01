#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import importlib.util
import os

# 默认配置文件路径
DEFAULT_CONFIG_FILE = Path(__file__).parent / "Diffuman" / "Hq_4_2_DNA_48.py"

# 尝试从配置文件加载参数
def load_config_values():
    """从 Hq_4_2_DNA_48.py 加载配置值"""
    config_path = os.getenv("DATA_PREP_CONFIG_FILE", str(DEFAULT_CONFIG_FILE))
    
    if not os.path.exists(config_path):
        # 如果配置文件不存在，返回默认值
        return get_default_values()
    
    try:
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 从配置文件读取值
        config_dict = getattr(config_module, 'config', {})
        data_prep_config = config_dict.get('data_preparation.py', config_dict.get('data_prep_config', {}))
        
        return {
            "subject": getattr(config_module, 'subject', 'Actor01'),
            "sequence": getattr(config_module, 'sequence', 'Sequence1'),
            "resolution": getattr(config_module, 'resolution', '1x'),
            "images_dir": Path(data_prep_config.get('images_dir', getattr(config_module, 'images_dir', ''))),
            "masks_dir": Path(data_prep_config.get('masks_dir', getattr(config_module, 'masks_dir', ''))),
            "transforms_json": Path(data_prep_config.get('transforms_json', getattr(config_module, 'transforms_json', ''))),
            "output_root": Path(data_prep_config.get('output_root', getattr(config_module, 'output_root', ''))),
            "smplx_root": Path(data_prep_config.get('smplx_root', getattr(config_module, 'smplx_root', ''))),
            "actorshq_smplx_zip": Path(data_prep_config.get('actorshq_smplx_zip', getattr(config_module, 'ActorsHQ_SMPLXzip', ''))) if data_prep_config.get('actorshq_smplx_zip') or getattr(config_module, 'ActorsHQ_SMPLXzip', None) else None,
            "actorshq_smplx_del_path": Path(data_prep_config.get('actorshq_smplx_del_path', getattr(config_module, 'ActorsHQ_SMPLXzip_del_path', ''))) if data_prep_config.get('actorshq_smplx_del_path') or getattr(config_module, 'ActorsHQ_SMPLXzip_del_path', None) else None,
            "masker_prompt": data_prep_config.get('masker_prompt', 'garment'),
            "gender": data_prep_config.get('gender', 'female'),
            "box_threshold": data_prep_config.get('box_threshold', 0.35),
            "text_threshold": data_prep_config.get('text_threshold', 0.25),
            "skip_masking": data_prep_config.get('skip_masking', False),
            "skip_reorg": data_prep_config.get('skip_reorg', False),
            "copy_data": data_prep_config.get('copy_data', False),
            "skip_smplx": data_prep_config.get('skip_smplx', True),
            "skip_json": data_prep_config.get('skip_json', False),
            "skip_cloth_extraction": data_prep_config.get('skip_cloth_extraction', False),
        }
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return get_default_values()

def get_default_values():
    """返回默认值（作为后备）"""
    return {
        "subject": "Actor01",
        "sequence": "Sequence1",
        "resolution": "1x",
        "images_dir": Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48/images_alpha"),
        "masks_dir": Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48/fmasks"),
        "transforms_json": Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48/transforms.json"),
        "output_root": Path("/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48/seg_clothes/Actor01/Sequence1"),
        "smplx_root": Path("/mnt/cvda/cvda_phava/dataset/body_models"),
        "masker_prompt": "garment",
        "gender": "female",
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "skip_masking": False,
        "skip_reorg": False,
        "copy_data": False,
        "skip_smplx": True,
        "skip_json": False,
        "skip_cloth_extraction": False,
    }

# 加载配置值
DATA_PREP_ARGS = load_config_values()

