#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/1 19:32
# @Author  : jc Han
# @help    :

from config.config import get_parser
from config import config
import argparse
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import importlib
import random
from datetime import datetime
import torch.nn.functional as F
from tqdm import tqdm
import sys
from utils.general_util import (
    safe_state,
    namespace_to_dict,
    dict_to_namespace,
    load_gs_attributes,
    load_pcd,
)
from utils.net_util import delete_batch_idx, to_cuda
from utils.other_util import save_images
from utils.log_util import *
from utils.render_utils import filter_large_scales
from utils.loss_util import l1_loss, ssim, psnr, total_loss
from models.gaussians.gaussian_model import GaussianModel
from models.gaussians.gaussian_renderer import render, Camera
from utils.covert.tools_view import cover_posed_to_ref
import os
from train import TrainRunner

try:
    import wandb

    print("✅ wandb 导入成功")
except Exception as e:
    print(f"❌ wandb 导入失败: {e}")
    
wandb = None
try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

if __name__ == "__main__":

    np.random.seed(2025)

    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--config", type=str, default="/mnt/cvda/cvda_phava/code/Han/PhyAvatar/config/DNARender/0013_01.yaml"
    )
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", type=bool, default=False)
    args = parser.parse_args()
    args = get_parser(args)

    if wandb is not None:
        try:
            print("🚀 正在初始化 wandb...")
            # 增加超时时间到120秒
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "PhyAvatar"),
                name=os.path.basename(args.mode),
                config={"supplement": args.supplement},
                settings=wandb.Settings(init_timeout=120),
            )
            print(
                f"✅ wandb 初始化成功！项目: {wandb.run.project}, 运行: {wandb.run.name}"
            )
            print(f"🌐 查看链接: {wandb.run.get_url()}")
        except Exception as e:
            print(f"❌ wandb 初始化失败: {e}")
            print("💡 尝试使用离线模式...")
            try:
                # 尝试离线模式
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project=os.getenv("WANDB_PROJECT", "PhyAvatar"),
                    name=os.path.basename(args.mode),
                    config={"supplement": args.supplement},
                    mode="offline",
                )
                print("✅ wandb 离线模式初始化成功！")
            except Exception as offline_e:
                print(f"❌ 离线模式也失败: {offline_e}")
                print("💡 请检查网络连接或使用 wandb login --relogin 重新登录")
                wandb = None
    else:
        print("⚠️  wandb 未安装或导入失败，跳过实验跟踪")

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    runner = TrainRunner(
        args=args,
        tp=args.train,
        opt=args.OptimizationParams,
        pipe=args.PipelineParamsNoparse,
        mp=args.ModelParams,
    )
    runner.run()
