#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
from scipy.spatial.transform import Rotation as R
import json
import torch
import numpy as np
import config
from utils.general_util import inverse_sigmoid, get_expon_lr_func, build_rotation
from plyfile import PlyData, PlyElement
from utils.sh_util import RGB2SH ,SH2RGB
from utils.graphics_util import BasicPointCloud
from torch import nn
import  math
from utils.log_util import plot_statistics
from simple_knn._C import distCUDA2
from utils.general_util import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R_scipy
import os
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass




class GaussianModel:

    def setup_functions(self):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree , optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self.xyz = torch.empty(0)
        self.features_dc = torch.empty(0)
        self.features_rest = torch.empty(0)
        self.scaling = torch.empty(0)
        self.rotation = torch.empty(0)
        self.opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self.xyz,
            self.features_dc,
            self.features_rest,
            self.scaling,
            self.rotation,
            self.opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self.xyz,
        self.features_dc,
        self.features_rest,
        self.scaling,
        self.rotation,
        self.opacity,
        self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom,
        self.opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(self.opt_dict)

    @property
    def get_max_sh_degree(self):
        return self.max_sh_degree

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self.features_dc

    @property
    def get_features_rest(self):
        return self.features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)
    @property
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1]*self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1]*self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def get_gaussians_properties(self, gaussian_model):

        xyz = gaussian_model.xyz
        opacity = gaussian_model.opacity
        scales = gaussian_model.scaling
        rotations = gaussian_model.rotation
        cov3D_precomp = None
        shs = None
        if gaussian_model.use_rgb:
            colors_precomp = gaussian_model.shs
        else:
            raise NotImplementedError

        return xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp

    def update_learning_rate(self, iteration):
        # 遍历优化器的所有参数组（每个组可包含不同参数和超参数）
        for param_group in self.optimizer.param_groups:
            # 检查参数组名称是否为 "xyz"（例如3D高斯点的位置参数）
            if param_group["name"] == "xyz":
                # 调用学习率调度器，传入当前迭代步数，计算学习率
                lr = self.xyz_scheduler_args(iteration)
                # 更新该参数组的学习率
                param_group['lr'] = lr
                # 返回当前学习率（可选，用于日志记录或调试）
                return lr

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense  # Initial density
        # 累积每个3D高斯点位置（xyz）的梯度幅值，用于后续判断是否需要分裂（split）或克隆（clone）高斯点。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 记录每个高斯点梯度累积的归一化分母（即梯度被累加的次数），用于计算梯度的移动平均值
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self.xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self.features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 连续的学习率衰减函数
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def save_ply(self, path,iteration):

        os.makedirs(os.path.dirname(path), exist_ok = True)
        path = os.path.join(path, f"{iteration }.ply")

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # f_dc = RGB2SH(self.features_dc)
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes() if
                      not attribute.startswith("f_rest_")]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def fetchPly(self, path):
        """
        从PLY文件读取点云数据，返回BasicPointCloud对象
        
        Args:
            path: PLY文件路径
            
        Returns:
            BasicPointCloud: 包含points, colors, normals的命名元组
        """
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        return BasicPointCloud(points=positions, colors=colors)

    def create_from_pcd(self, points, colors, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(np.asarray(points))
        if not isinstance(colors, torch.Tensor):
            colors = torch.tensor(np.asarray(colors))

        fused_point_cloud = points.float().cuda()
        # fused_color = RGB2SH(colors.float().cuda())
        fused_color = colors.float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color.squeeze(-1)
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self.xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.scaling = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def load_ply(self, path, trans_json_path=None):
        # 1. --- LOAD RAW DATA FROM PLY ---
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # [FIX for ValueError]: Handle max_sh_degree = 0 case
        if self.max_sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_extra = features_extra.reshape((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        else:
            features_extra = np.empty(shape=(xyz.shape[0], 3, 0))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Convert final, transformed NumPy arrays to PyTorch tensors
        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.spatial_lr_scale = 1
        
        print("[load_ply] PLY data loading and all transformations are complete.")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print(torch.all(viewspace_point_tensor.grad.max()),self.xyz_gradient_accum[update_filter].max(),self.denom[update_filter].max())
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)

        self.denom[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)  # 对梯度高于 max_grad 且尺寸较小的高斯点进行克隆，在周围增加新点以捕捉细节
        self.densify_and_split(grads, max_grad, extent)  # 对梯度高于 max_grad 且尺寸较大的高斯点进行分裂，将其拆分为多个小高斯点


        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            # 若高斯点在屏幕空间（max_radii2D）或世界空间（get_scaling）的尺寸超过阈值，则标记为待修剪。
            big_points_vs = self.max_radii2D > max_screen_size  # 屏幕空间尺寸过大
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # 世界空间尺寸过大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)

        self.xyz = optimizable_tensors["xyz"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.opacity = optimizable_tensors["opacity"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        # 0.01 * 2.9 = 0.029


        new_xyz = self.xyz[selected_pts_mask]
        new_features_dc = self.features_dc[selected_pts_mask]
        new_features_rest = self.features_rest[selected_pts_mask]
        new_opacities = self.opacity[selected_pts_mask]
        new_scaling = self.scaling[selected_pts_mask]
        new_rotation = self.rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_tmp_radii)
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.xyz = optimizable_tensors["xyz"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.opacity = optimizable_tensors["opacity"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def reset_sacles(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors