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

try:
    import wandb

    print("✅ wandb 导入成功")
except Exception as e:
    print(f"❌ wandb 导入失败: {e}")
    wandb = None

dens_statistic_dict = {
    "n_points_cloned": 0,
    "n_points_split": 0,
    "n_points_mercied": 0,
    "n_points_pruned": 0,
    "redundancy_threshold": 0,
    "opacity_threshold": 0,
}

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


def load_checkpoint(gaussians, model_path, opt, sh_degree=0):
    # Find checkpoint

    (checkpoint_data, iteration) = torch.load(args.train.save_path + "/best_chkpnt.pth")

    checkpt_dir = os.path.join(model_path, "result")

    gs_path = os.path.join(checkpt_dir, f"{iteration}")

    gaussians.load_ply(gs_path + ".ply")
    gaussians.training_setup(opt)

    gaussians.restore(checkpoint_data, opt)

    print(f" Load {iteration} model checkpoint ")
    return iteration


def training(args):
    mp = args.ModelParams
    opt = args.OptimizationParams
    pipe = args.PipelineParamsNoparse

    AvatarDataset = importlib.import_module("dataset.get_dataset").__getattribute__(
        args.train.dataset
    )

    train_data = namespace_to_dict(args.train.data)

    dataset = AvatarDataset(**train_data)
    first_iter = 0

    # load gaussian
    gaussians = GaussianModel(mp.max_sh_degree, opt.optimizer_type)
    # init_points = torch.tensor(load_pcd(args.ref_gs_path)['points'])
    # gaussians.create_from_pcd(init_points,torch.rand_like(init_points),spatial_lr_scale = 2.5)

    gaussians.load_ply(args.ref_gs_path)
    gaussians.training_setup(opt)

    if args.train.checkpoint != "":
        first_iter = load_checkpoint(gaussians, args.train.save_path, opt)

    background = (
        torch.from_numpy(np.asarray([0, 0, 0])).to(torch.float32).to(config.device)
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    args.train.min_l1_loss = 999

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        idx = iteration % len(args.train.data.used_cam_ids)

        items = dataset.getitem(idx)

        items = delete_batch_idx(items)

        items = to_cuda(items)

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Render
        background_color = (
            torch.rand((3), device=config.device)
            if opt.random_background
            else background
        )

        render_result = render(
            gaussians,
            pipe,
            Camera.from_c2w(
                items["extr"],
                items["intr"],
                items["img_h"],
                items["img_w"],
                items["img_name"],
                items["ty"],
            ),
            background_color,
        )

        (
            comp_rgb,
            comp_rgb_bg,
            comp_mask,
            comp_depth,
            radii,
            visibility_filter,
            viewspace_points,
        ) = (
            render_result["comp_rgb"],
            render_result["comp_rgb_bg"],
            render_result["comp_mask"],
            render_result["comp_depth"],
            render_result["radii"],
            render_result["visibility_filter"],
            render_result["viewspace_points"],
        )

        gt_image = items["color_img"].permute(2, 0, 1)
        gt_mask = items["mask_img"].float().unsqueeze(0)

        loss, _ = total_loss(comp_mask, gt_mask, comp_rgb, gt_image, gaussians, args)

        # comp_rgb, gt_image = crop_image(gt_mask.squeeze(0), patch_size, random_patch_flag, background_color, comp_rgb, gt_image,)

        loss.backward()

        iter_end.record()

        with torch.no_grad():

            training_report(
                args,
                pipe,
                progress_bar,
                iteration,
                gaussians,
                render,
                dataset,
                background_color,
                loss,
            )

            if iteration % opt.scale_reset_interval == 0:

                gaussians.max_radii2D = gaussians.max_radii2D.to(config.device)

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )

                filtered_visibility = filter_large_scales(gaussians, visibility_filter)

                visible_indices = torch.where(filtered_visibility)[0]

                max_scales = (
                    torch.max(gaussians.get_scaling, dim=1)
                    .values.detach()
                    .cpu()
                    .numpy()
                )
                opacities = gaussians.get_opacity.cpu().numpy()
                # plot_statistics_with_annotations(max_scales, opacities, save_path='/mnt/cvda/cvda_phava/code/Han/LHM/train_data/custom_motion/custom_dress/gaussian_statistics.png')

            # Densification
            if opt.reduce_gs:
                if iteration < opt.densify_until_iter:

                    gaussians.max_radii2D = gaussians.max_radii2D.to(config.device)
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )

                    gaussians.add_densification_stats(
                        viewspace_points, visibility_filter
                    )

                    if (
                        iteration < opt.densify_final_iter
                        and iteration % opt.densification_interval == 0
                    ):

                        # 此时，过大的高斯点会被分裂（避免单个高斯点覆盖太大区域，导致渲染模糊）
                        size_threshold = (
                            5 if iteration > opt.opacity_reset_interval else None
                        )

                        gaussians.densify_and_prune(
                            args,
                            iteration,
                            opt.densify_grad_threshold,  # 梯度阈值（控制分裂）
                            0.05,  # 最小不透明度阈值（控制修剪） 0.005
                            items["radius"],  # 场景尺度（用于标准化）
                            size_threshold,  # 大小阈值（控制分裂）
                            radii,
                            dens_statistic_dict,
                        )

                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

                elif (
                    args.prune_dead_points
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.prune(1 / 255, items["radius"], None, dens_statistic_dict)
            else:

                if iteration < opt.densify_until_iter:

                    gaussians.max_radii2D = gaussians.max_radii2D.to(config.device)
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )

                    gaussians.add_densification_stats(
                        viewspace_points, visibility_filter
                    )

                    if (
                        iteration < opt.densify_final_iter
                        and iteration % opt.densification_interval == 0
                    ):

                        # 此时，过大的高斯点会被分裂（避免单个高斯点覆盖太大区域，导致渲染模糊）
                        size_threshold = (
                            5 if iteration > opt.opacity_reset_interval else None
                        )

                        gaussians.densify_and_prune(
                            args,
                            iteration,
                            opt.densify_grad_threshold,  # 梯度阈值（控制分裂）
                            0.1,  # 最小不透明度阈值（控制修剪） 0.005
                            items["radius"],  # 场景尺度（用于标准化）
                            size_threshold,  # 大小阈值（控制分裂）
                            radii,
                        )

                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                progress_bar.close()

            if wandb is not None and wandb.run is not None and iteration % 10 == 0:
                try:
                    wandb.log({"iteration": iteration})
                except Exception:
                    pass

    if opt.reduce_gs:
        gaussians.produce_clusters(store_dict_path=args.train.save_path)
        gaussians.save_ply(iteration, quantise=True)
        gaussians.save_ply(iteration, quantise=True, half_float=True)
    if wandb is not None and wandb.run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def training_report(
    args, pipe, progress_bar, iteration, gaussian, renderFunc, dataset, bg, train_loss
):

    views_num = len(args.train.data.used_cam_ids)

    test_loss = 0.0
    L1 = 0.0
    ellipsoid_loss = 0.0
    lpips_loss = 0.0
    L1_mask = 0.0
    Lssim = 0.0
    for idx in range(views_num):
        items = dataset.getitem(idx)
        items = to_cuda(items)

        render_result = renderFunc(
            gaussian,
            pipe,
            Camera.from_c2w(
                items["extr"],
                items["intr"],
                items["img_h"],
                items["img_w"],
                items["img_name"],
                items["ty"],
            ),
            bg,
        )

        comp_rgb = render_result["comp_rgb"]
        comp_mask = render_result["comp_mask"]
        gt_image = items["color_img"].permute(2, 0, 1)
        gt_mask = items["mask_img"].float().unsqueeze(0)

        # save
        comp_rgb_mask = comp_rgb * gt_mask

        images = {
            "comp_rgb": comp_rgb,
            "gt_rgb": gt_image,
            "comp_rgb_mask": comp_rgb_mask,
            "comp_mask": comp_mask,
        }

        save_images(args.train.result_path, images, items["img_name"], iteration)

        gaussian.save_ply(f"{args.train.result_path}", iteration)

        loss, dict_loss = total_loss(
            comp_mask, gt_mask, comp_rgb, gt_image, gaussian, args
        )

        test_loss += loss
        L1 += dict_loss["L1"]
        ellipsoid_loss += dict_loss["ellipsoid_loss"]
        lpips_loss += dict_loss["lpips_loss"]
        L1_mask += dict_loss["L1_mask"]
        Lssim += dict_loss["Lssim"]

    test_loss /= views_num
    L1 /= views_num
    ellipsis /= views_num
    lpips_loss /= views_num
    L1_mask /= views_num
    Lssim /= views_num

    try:
        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "total_loss": test_loss.detach().item(),
                    "L1": L1.detach().item(),
                    "ellipsoid_loss": ellipsis.detach().item(),
                    "lpips_loss": lpips_loss.detach().item(),
                    "L1_mask": L1_mask.detach().item(),
                    "Lssim": Lssim.detach().item(),
                }
            )
    except Exception:
        pass

    log_info = "\n [ITER {}] : test_loss {} train_L1 {}  min_loss {} time {} ".format(
        iteration, test_loss, train_loss, args.train.min_l1_loss, args.mode
    )

    if args.train.min_l1_loss > test_loss:
        args.train.min_l1_loss = test_loss
        print(log_info)
        torch.save(
            (gaussian.capture(), iteration), args.train.save_path + "/best_chkpnt.pth"
        )

        if iteration > 100:
            comp_rgb_mask = comp_rgb * gt_mask

            images = {
                "comp_rgb": comp_rgb,
                "gt_rgb": gt_image,
                "comp_rgb_mask": comp_rgb_mask,
                "comp_mask": comp_mask,
            }

            save_images(args.train.result_path, images, items["img_name"], iteration)
            if args.train.prune_dead_points:
                gaussian.prune(1 / 255, items["radius"], None, dens_statistic_dict)
            gaussian.save_ply(f"{args.train.result_path}", iteration)

    if iteration % 10 == 0:
        # progress_bar.set_postfix({'Loss': f"{test_loss:.4f}",'min_loss': f"{args.train.min_l1_loss:.4f}"})
        progress_bar.update(10)

    with open(args.train.log_path, "a", encoding="utf-8") as f:
        f.write(f"{log_info}")
        if iteration == args.train.saving_iterations[-1]:
            f.write(f"Mini loss {args.train.min_l1_loss} ")

    torch.cuda.empty_cache()


if __name__ == "__main__":

    np.random.seed(2025)

    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--config", type=str, default="./config/gs_optimize_gtmask_T_L1mask_T.yaml"
    )
    parser.add_argument(
        "--ref_gs_path",
        type=str,
        default="/mnt/cvda/cvda_phava/code/Han/LHM/gs_result/2025-08-28 02:03:13/0_0_1756317796.ply",
    )
    # parser.add_argument("--ref_gs_path", type=str, default='/mnt/cvda/cvda_phava/code/Han/PhyAvatar/utils/covert/result.ply')
    parser.add_argument("--output_path", type=str, default="./gs_result")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", type=bool, default=True)
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

    args.train.save_path = os.path.join(args.train.save_path, f"{args.mode}")
    os.makedirs(args.train.save_path, exist_ok=True)

    args.train.log_path = os.path.join(args.train.save_path, "log.txt")
    args.train.result_path = os.path.join(args.train.save_path, "result")

    os.makedirs(args.train.result_path, exist_ok=True)

    # transforming camera perspectives, from posed to ref
    print("Transforming camera perspectives, from posed to ref...")

    # cover_posed_to_ref(args.posted_gs_path, args.ref_gs_path)

    # ref_gs =load_gs_attributes(args.ref_gs_path)
    #
    # ref_gs = dict_to_namespace(ref_gs)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(args)
