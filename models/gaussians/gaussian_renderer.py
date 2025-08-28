#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from models.gaussians.gaussian_model import GaussianModel
import numpy as np
from utils.sh_util import eval_sh,SH2RGB

def auto_repeat_size(tensor, repeat_num, axis=0):
    repeat_size = [1] * tensor.dim()
    repeat_size[axis] = repeat_num
    return repeat_size


def aabb(xyz):
    return torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values


def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


def generate_rotation_matrix_y(degrees):
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y

def is_identity(matrix):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:  # 检查是否是方阵
        return False
    n = matrix.shape[0]
    return np.allclose(matrix, np.eye(n))

class Camera:
    def __init__(
        self,
        w2c,
        intrinsic,
        FoVx,
        FoVy,
        height,
        width,
        view_id,
        ty  ,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)
        self.ty =ty
        self.zfar = 100.0
        self.znear = 0.01
        self.view_id=view_id
        self.trans = trans
        self.scale = scale

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(w2c.device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = intrinsic

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width,view_id,ty):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(
            intrinsic,
            w=width,
            h=height
        )
        return Camera(
            w2c=w2c,
            intrinsic=intrinsic,
            FoVx=FoVx,
            FoVy=FoVy,
            height=height,
            width=width,
            view_id=view_id,
            ty=ty
        )


def render(
    gs: GaussianModel,
    pipe,
    viewpoint_camera: Camera,
    background_color,
    scaling_modifier=1.0,
    override_color=None,
    separate_sh=False,

):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (torch.zeros_like(gs.get_xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    bg_color = background_color
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform.float(),
        sh_degree=gs.get_max_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gs.get_xyz
    means2D = screenspace_points
    opacity = gs.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = gs.get_covariance()
    else:
        scales = gs.get_scaling
        rotations = gs.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # dc（基础颜色）与高阶球谐系数（shs）结合，根据视角和光照方向实时计算颜色
    shs = None
    colors_precomp = None # 最终RGB颜色  静态颜色
    dc =None # 基础漫反射颜色
    #
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = gs.get_features.transpose(1, 2).view(-1, 3, (gs.max_sh_degree+1)**2)
            dir_pp = (gs.get_xyz - viewpoint_camera.camera_center.repeat(gs.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gs.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = gs.get_features_dc, gs.get_features_rest
            else:
                colors_precomp = SH2RGB(gs.get_features_dc).squeeze(1).clamp(0,1).float()
                # shs = gs.get_features
                # colors_precomp = SH2RGB(gs.get_features_dc).squeeze(1).clamp(0,1)

    else:
        colors_precomp = override_color

    # print("colors_precomp.shape,colors_precomp.max()",colors_precomp.shape,colors_precomp.max(),colors_precomp.min())
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # NOTE that dadong tries to regress rgb not shs

    with torch.autocast(device_type=colors_precomp.device.type, dtype=torch.float32):
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D.float(),
            means2D=means2D.float(),
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity.float(),
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )
    # print(" rendered_image", rendered_image.shape, rendered_image.device)
    ret = {
        "comp_rgb": rendered_image[[2, 1, 0], ...].clamp(0, 1),  # BGR→RGB [3, H, W]
        "comp_rgb_bg": bg_color,
        "comp_mask": rendered_alpha.clamp(0, 1),
        "comp_depth": rendered_depth,
        "radii":radii, #  高斯点在屏幕空间的半径 # (N,)
        "visibility_filter": radii > 0, # 可见性过滤
        "viewspace_points": screenspace_points
    }

    return ret