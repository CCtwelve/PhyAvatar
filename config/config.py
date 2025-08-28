#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/2 17:50
# @Author  : jc Han
# @help    :

import argparse
import yaml
import os
import shutil
import torch
device = torch.device('cuda:5')

from types import SimpleNamespace


def get_parser(args_cfg, config_dict=None):
    """
    返回支持点号访问的嵌套配置对象（如 args.train.data）
    """
    # 加载YAML配置（原逻辑保持不变）
    if config_dict is None:
        with open(args_cfg.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

    # 保存原始默认值
    original_defaults = {
        k: v for k, v in vars(args_cfg).items()
        if not k.startswith('_') and not isinstance(v, type('Config', (), {}))
    }

    # 递归将字典转换为 SimpleNamespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        else:
            return d

    # 合并配置和默认值
    merged_dict = {**original_defaults, **config_dict}

    # 转换为嵌套的 SimpleNamespace 对象
    return dict_to_namespace(merged_dict)


def save_config(args,target_path):

    source_file=args.config
    file_name = os.path.basename(source_file)
    target_file = os.path.join(target_path, file_name)
    shutil.copyfile(source_file, target_file)
    print(f'File copied from {source_file} to {target_file}')