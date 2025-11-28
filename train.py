#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/1 19:32
# @Author  : jc Han
# @help    :

from config.config import get_parser
from config import config
import argparse
import torch
import lpips
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
import numpy as np
import importlib
import random
from datetime import datetime
import torch.nn.functional as F
from tqdm import tqdm
import sys
from utils.general_util import (

    namespace_to_dict,
    get_expon_lr_func,
)
from utils.net_util import delete_batch_idx, to_cuda
from utils.log_util import *
from utils.loss_util import l1_loss, ssim, psnr, calculate_ema_losses_and_string, CombinedLoss, refine_total_loss
from models.gaussians.gaussian_model import GaussianModel
from models.gaussians.gaussian_renderer import render, Camera
from utils.covert.tools_view import cover_posed_to_ref
import os
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

import wandb


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


def load_checkpoint(gaussians, save_path, opt):
    # Find checkpoint

    (checkpoint_data, iteration) = torch.load(save_path + "/best_chkpnt.pth")

    checkpt_dir = os.path.join(save_path, "result")

    gs_path = os.path.join(checkpt_dir, f"{iteration}")

    gaussians.load_ply(gs_path + ".ply")
    
    gaussians.training_setup(opt)

    gaussians.restore(checkpoint_data, opt)

    print(f" Load {iteration} model best_chkpnt.pth........... ")
    return iteration

class TrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.args = kwargs['args']
        self.tps = kwargs['tp']
        self.opt = kwargs['opt']
        self.pipe = kwargs['pipe']
        self.mp = kwargs['mp']
        
        # load dataset
        print("loading dataset...")
        AvatarDataset = importlib.import_module("dataset.get_dataset").__getattribute__(
            self.tps.dataset
        )
        train_data = namespace_to_dict(self.tps.data)
        self.dataset = AvatarDataset(**train_data)

        # create log directory
        self.save_path,self.log_path,self.result_path = create_log_dir(self.tps.save_path,self.args,self.args.config)
        
        # Initialize logging systems (wandb, tensorboard, plotting)
        initialize_logging(self.tps,self.args,self.save_path)
        
        # load gaussian
        self.gaussians = GaussianModel(self.mp.max_sh_degree, self.opt.optimizer_type)

        # init_points = torch.tensor(load_pcd(args.ref_gs_path)['points'])
        
        if self.args.train_type == "random":
            size = self.args.point_size
            init_points = torch.rand(size,3) * 2 -1
            radius = self.dataset.get_radius()
            self.gaussians.create_from_pcd(init_points,torch.rand_like(init_points),spatial_lr_scale = radius)

        elif self.args.train_type == "pcd":
            radius = self.dataset.get_radius()
            basePoints = self.gaussians.fetchPly(self.args.ref_gs_path)
            self.gaussians.create_from_pcd(basePoints.points,basePoints.colors,spatial_lr_scale = radius)   

        elif self.args.train_type == "gs" :
            self.gaussians.load_ply(self.args.ref_gs_path,trans_json_path = getattr(self.args,"trans_json_path",None)) 

        self.gaussians.training_setup(self.opt)
        
        if self.tps.checkpoint == True:
            self.first_iter = load_checkpoint(self.gaussians, self.save_path, self.opt)
        else:
            self.first_iter = 0

        self.background = (
            torch.from_numpy(np.asarray([0, 0, 0])).to(torch.float32).to(config.device)
        )
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        self.use_sparse_adam = self.opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

        self.tps.min_l1_loss = 999

        self.total_mask_loss_fn = CombinedLoss(weight_bce=0.5, weight_dice=0.5)

        self.lpips_net = lpips.LPIPS(net='vgg', spatial=True).to(config.device)
        self.lpips_net.eval() 
    def run(self):
        self.progress_bar = tqdm(
            range(self.first_iter, self.opt.iterations),
            desc="Training progress",
            leave=False,
            dynamic_ncols=False,
            disable=self.args.quiet,
        )
        if not isinstance(self.tps.data.used_cam_ids, list):
            self.tps.data.used_cam_ids = range(self.tps.data.used_cam_ids)


        self.first_iter += 1
        for iteration in range(self.first_iter, self.opt.iterations + 1):

            idx  = random.randint(0, len(self.tps.data.used_cam_ids) - 1) 

            items = self.dataset.getitem(idx)

            items = delete_batch_idx(items)

            items = to_cuda(items)

            self.iter_start.record() 

            self.gaussians.update_learning_rate(iteration)

            self.background_color = (
                torch.rand((3), device=config.device)
                if self.opt.random_background
                else self.background 
            )
                
            render_result = render(self.gaussians,self.pipe,Camera.from_c2w(items["extr"],items["intr"],items["img_h"],items["img_w"],items["img_name"]),self.background_color)

            render_rgb,render_rgb_bg,render_alpha,render_depth,radii,visibility_filter,viewspace_points = render_result["render_rgb"],render_result["render_rgb_bg"],render_result["render_alpha"],render_result["render_depth"],render_result["radii"],render_result["visibility_filter"],render_result["viewspace_points"]

            gt_image = items["color_img"].permute(2, 0, 1)
            
            gt_mask = items["mask_img"].float().unsqueeze(0)

            loss, dict_loss = refine_total_loss(self.args,iteration,self.tps,render_alpha, gt_mask, render_rgb, gt_image, self.gaussians,self.total_mask_loss_fn,self.lpips_net,items["radius"],self.opt.percent_dense)

            loss.backward()

            self.iter_end.record()

            with torch.no_grad():

                # Calculate EMA losses and generate loss string using utility function
                loss_str = calculate_ema_losses_and_string(dict_loss, iteration)
                
                # Save all smoothed loss components to log file
                append_to_log(self.log_path, loss_str)

                if iteration % 10 == 0:
                    self.progress_bar.set_postfix({"Loss": f" {loss_str} "})
                    self.progress_bar.update(10)

                if iteration == self.opt.iterations:
                    self.progress_bar.close()

                self.training_report(iteration, loss, dict_loss)

                # Densification  
                if iteration < self.opt.iterations_interval_list[-1] :
                    # self.gaussians.max_radii2D = self.gaussians.max_radii2D.to(config.device)
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],radii[visibility_filter])

                    self.gaussians.add_densification_stats(viewspace_points, visibility_filter)

                    if iteration > self.opt.densify_from_iter and  iteration % self.opt.densification_interval == 0:
                        size_threshold = self.opt.size_threshold if iteration > self.opt.opacity_reset_interval else None

                        self.update_params_by_stage(iteration)

                        if iteration >= self.opt.densify_until_iter:
                            if_clone_split = False
                        else:
                            if_clone_split = True

                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold,  # Gradient threshold (controls splitting)
                            self.opt.min_opacity,  # Minimum opacity threshold (controls pruning) 
                            items["radius"],  # Scene scale (for normalization)
                            size_threshold,  # Size threshold (controls splitting)
                            radii,
                        )

                if iteration % self.opt.opacity_reset_interval == 0 and iteration < self.opt.iterations_interval_list[1]:
                    self.gaussians.reset_opacity()


                # Optimizer step
                if iteration < self.opt.iterations:

                    if self.use_sparse_adam:
                        visible = radii > 0
                        self.gaussians.optimizer.step(visible, radii.shape[0])
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    self.progress_bar.close()

        # Clean up logging systems
        cleanup_logging(self.tps)

    def training_report(self, iteration,loss,dict_loss):
        # add loss to tb writer
        tb_writer = self.tps.tb_writer
        if tb_writer is not None:
            tb_writer.add_scalar("train_loss/total_loss", loss, iteration)
            tb_writer.add_scalar("train_loss/L1", dict_loss["L1"], iteration)
            tb_writer.add_scalar("train_loss/ellipsoid_loss", dict_loss["ellipsoid_loss"], iteration)
            tb_writer.add_scalar("train_loss/lpips_loss", dict_loss["lpips_loss"], iteration)
            tb_writer.add_scalar("train_loss/ssim", dict_loss["ssim"], iteration)
            if "opacity_loss" in dict_loss:
                tb_writer.add_scalar("train_loss/opacity_loss", dict_loss["opacity_loss"], iteration)
            if "combined_mask_loss" in dict_loss:
                tb_writer.add_scalar("train_loss/combined_mask_loss", dict_loss["combined_mask_loss"], iteration)
            if "l1_mask_loss" in dict_loss:
                tb_writer.add_scalar("train_loss/l1_mask_loss", dict_loss["l1_mask_loss"], iteration)

        images_list= []
        # Save image and ply every 1000 iterations
        if iteration % self.opt.log_save_interval == 0:

            views_num_train = len(self.tps.data.used_cam_ids)

            views_num_test = len(self.tps.data.test_cam_ids)

            test_loss = 0.0
            L1 = 0.0
            ellipsoid_loss = 0.0
            lpips_loss = 0.0
            l1_mask_loss = 0.0
            ssim = 0.0
            combined_mask_loss = 0.0
            opacity_loss = 0.0
            psnr_val = 0.0
            scale_loss = 0.0
            for idx in range(views_num_train):    
                items = self.dataset.getitem(idx,record=True)
                items = to_cuda(items)

                render_result = render(self.gaussians,self.pipe,Camera.from_c2w(items["extr"],items["intr"],items["img_h"],items["img_w"],items["img_name"]),self.background_color,)

                render_rgb = render_result["render_rgb"]
                render_alpha = render_result["render_alpha"]
                

                gt_image = items["color_img"].permute(2, 0, 1)
                gt_mask = items["mask_img"].float().unsqueeze(0)

                gt_mask_np = gt_mask.squeeze().cpu().detach().numpy()
                render_alpha_np = render_alpha.squeeze().cpu().detach().numpy()

                inverted_mask_np = create_mask_overlay(gt_mask_np, render_alpha_np)

                inverted_mask = torch.from_numpy(inverted_mask_np).permute(2, 0, 1)

                images = {
                    "render_rgb": render_rgb,
                    "gt_rgb": gt_image,
                    # "render_rgb_mask": render_rgb * gt_mask,
                    "render_mask": render_alpha,
                    "gt_mask":gt_mask,
                    "inverted_mask":inverted_mask
                    # "inverted_mask": gt_mask * render_alpha,
                    
                }

                images_list.append(images)

                if getattr(self.opt,"is_save_location",False):
                    save_images(self.result_path, images, items["img_name"], iteration)
                
                loss, dict_loss = refine_total_loss(self.args,iteration,self.tps,render_alpha, gt_mask, render_rgb, gt_image, self.gaussians, self.total_mask_loss_fn,self.lpips_net,items["radius"],self.opt.percent_dense)

                test_loss += loss
                L1 += dict_loss["L1"]
                ellipsoid_loss += dict_loss["ellipsoid_loss"]
                lpips_loss += dict_loss["lpips_loss"]
                l1_mask_loss += dict_loss["l1_mask_loss"]
                combined_mask_loss += dict_loss["combined_mask_loss"]
                ssim += dict_loss["ssim"]
                opacity_loss += dict_loss["opacity_loss"]
                scale_loss += dict_loss["scale_loss"]
                
                # Calculate PSNR
                render_rgb_normalized = render_rgb * gt_mask
                gt_image_normalized = gt_image * gt_mask
                psnr_val += psnr(render_rgb_normalized, gt_image_normalized)
                if tb_writer is not None:
                    tb_writer.add_images("train_view_{}/render_rgb".format(items["img_name"]), images["render_rgb"], global_step=iteration, dataformats='CHW')
                    tb_writer.add_images("train_view_{}/render_alpha".format(items["img_name"]), torch.clip(images["render_mask"],0,1), global_step=iteration, dataformats='CHW')
                    tb_writer.add_images("train_view_{}/render_alpha_mask".format(items["img_name"]), (1 - gt_mask) * render_alpha, global_step=iteration, dataformats='CHW')
                    tb_writer.add_images("train_view_{}/gt".format(items["img_name"]), gt_image, global_step=iteration, dataformats='CHW')

            test_loss /= views_num_train
            L1 /= views_num_train
            ellipsoid_loss /= views_num_train
            lpips_loss /= views_num_train
            l1_mask_loss /= views_num_train
            combined_mask_loss /= views_num_train
            ssim /= views_num_train
            psnr_val /= views_num_train
            opacity_loss /= views_num_train
            scale_loss /= views_num_train
            # --------------------
            # Record and optional log/plot
            # --------------------
            # Log training metrics to all configured logging systems
            log_training_metrics(self.tps, iteration, test_loss, L1, ellipsoid_loss, lpips_loss, l1_mask_loss, ssim, psnr_val,opacity_loss,scale_loss,self.args.mode)

            log_info = "[ITER {}] : test_loss {}  min_loss {} l1_mask_loss {} combined_mask_loss {} opacity_loss {} PSNR {} SSIM {} mode{}".format(
                iteration, test_loss, self.tps.min_l1_loss, l1_mask_loss, combined_mask_loss, opacity_loss, psnr_val, ssim, scale_loss, self.args.mode        
            )
            print(log_info)
            append_to_log(self.log_path, log_info)
            # test view
            for idx in range(views_num_test):
                items = self.dataset.getitem(idx, mode="test")
                items = to_cuda(items)
                render_result = render(self.gaussians,self.pipe,Camera.from_c2w(items["extr"],items["intr"],items["img_h"],items["img_w"],items["img_name"]),self.background_color,)
                render_rgb = render_result["render_rgb"]
                render_alpha = render_result["render_alpha"]
                gt_image = items["color_img"].permute(2, 0, 1)
                gt_mask = items["mask_img"].float().unsqueeze(0)
                images = {
                    "render_rgb": render_rgb,
                    "gt_rgb": gt_image,
                    "render_rgb_mask": render_rgb * gt_mask,
                    "render_alpha": render_alpha,
                    "inverted_mask": (1 - gt_mask) * render_alpha,
                }

                # if getattr(self.opt,"is_save_location",False):
                #     save_images(self.result_path, images, items["img_name"], iteration)
                if tb_writer is not None:
                    tb_writer.add_images("test_{}/render_rgb".format(items["img_name"]), images["render_rgb"], global_step=iteration, dataformats='CHW')
                    tb_writer.add_images("test_{}/render_alpha".format(items["img_name"]), torch.clip(images["render_alpha"],0,1), global_step=iteration, dataformats='CHW')
                    tb_writer.add_images("test_{}/render_alpha_mask".format(items["img_name"]), (1 - gt_mask) * render_alpha, global_step=iteration, dataformats='CHW')
                    tb_writer.add_images("test_{}/gt".format(items["img_name"]),gt_image, global_step=iteration, dataformats='CHW')
            self.gaussians.save_ply(f"{self.result_path}", iteration)
            


            if self.tps.min_l1_loss > test_loss:
                self.tps.min_l1_loss = test_loss
                torch.save((self.gaussians.capture(), iteration), self.save_path + "/best_chkpnt.pth")  
            
            torch.save((self.gaussians.capture(), iteration), self.save_path + "/last_chkpnt.pth")  

        torch.cuda.empty_cache()

    def update_params_by_stage(self, iteration: int) -> float:

        if not getattr(self.args, "is_stage", False):
            pass
        else:
            prune_intervals = self.opt.iterations_interval_list

            if iteration < prune_intervals[0]:
                self.opt.min_opacity = self.opt.min_opacity_interval_list[0]
                self.opt.lamda_l1mask = self.args.train.loss.lamda_l1mask_list[0]
                self.opt.densify_grad_threshold = self.opt.densify_grad_threshold_list[0]
                self.opt.opacity_reset_interval = self.opt.opacity_reset_interval_list[0]
            
            elif iteration < prune_intervals[1] and iteration > prune_intervals[0]:
                self.opt.min_opacity  = self.opt.min_opacity_interval_list[1]
                self.opt.lamda_l1mask = self.args.train.loss.lamda_l1mask_list[1]
                self.opt.densify_grad_threshold = self.opt.densify_grad_threshold_list[1]
                self.opt.opacity_reset_interval = self.opt.opacity_reset_interval_list[1]

            elif iteration < prune_intervals[2] and iteration > prune_intervals[1]:
                self.opt.min_opacity = self.opt.min_opacity_interval_list[2]
                self.opt.lamda_l1mask = self.args.train.loss.lamda_l1mask_list[2]
                self.opt.densify_grad_threshold = self.opt.densify_grad_threshold_list[2]
                self.opt.opacity_reset_interval = self.opt.opacity_reset_interval_list[2]

            elif len(prune_intervals) > 3 and iteration > prune_intervals[2]:
                self.opt.min_opacity = self.opt.min_opacity_interval_list[3]
