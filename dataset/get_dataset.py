#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/3 22:54
# @Author  : jc Han
# @help    :
import glob
import json
import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation
class DatasetBase(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        test_cam_ids = None,
    ):
        super(Dataset, self).__init__()
        print(data_dir,frame_range)
        self.data_dir = data_dir

        print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
        frame_range = range(frame_range[0], frame_range[1], frame_range[2])
        self.pose_list = list(frame_range)

        if isinstance(used_cam_ids, list):
            self.used_cam_ids = used_cam_ids
        else:
            self.used_cam_ids = range(0,used_cam_ids)

        if test_cam_ids is None:
            self.test_cam_ids = []
        else:
            self.test_cam_ids = test_cam_ids

        print('# Used camera ids: ', self.used_cam_ids)

        self.train_data_list = []
        for pose_idx in self.pose_list:
            for view_idx in self.used_cam_ids:
                self.train_data_list.append((pose_idx, view_idx))

        self.test_data_list = []
        for pose_idx in self.pose_list:
            for view_idx in self.test_cam_ids:
                self.test_data_list.append((pose_idx, view_idx))

        self.load_cam_data()

        self.getNerfppNorm()

    def __len__(self):
        return len(self.train_data_list)

    def getitem(self, index, mode="train",record=False):
        if mode == "train":
            pose_idx, view_idx = self.train_data_list[index]
        else:
            if len(self.test_cam_ids) == 0:
                return []
            else:
                pose_idx, view_idx = self.test_data_list[index]

        data_idx = (pose_idx, view_idx)

        data_item = dict()
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx

        color_img, mask_img = self.load_color_mask_images(pose_idx, view_idx)

        color_img = (color_img / 255.).astype(np.float32)

        if record:
            print(pose_idx, view_idx,self.extr_mats[view_idx],self.cam_names[view_idx])

        data_item.update({
            'img_name': self.cam_names[view_idx],
            'img_h': color_img.shape[0],
            'img_w': color_img.shape[1],
            'extr': self.extr_mats[view_idx],
            'intr': self.intr_mats[view_idx],
            'color_img': color_img,
            'mask_img': mask_img,
            'radius':self.radius,
        })

        return data_item

    def get_radius(self):
        return self.radius

    @staticmethod
    def get_boundary_mask(mask, kernel_size = 5):
        """
        :param mask: np.uint8
        :param kernel_size:
        :return:
        """
        mask_bk = mask.copy()
        thres = 128
        mask[mask < thres] = 0
        mask[mask > thres] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv.erode(mask.copy(), kernel)
        mask_dilate = cv.dilate(mask.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        boundary_mask = np.logical_or(boundary_mask,
                                      np.logical_and(mask_bk > 5, mask_bk < 250))

        return boundary_mask, mask == 1

class DatasetActorsHQ(DatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        test_cam_ids = None,
    ):
        super(DatasetActorsHQ, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            test_cam_ids,
        )

    def load_color_mask_images(self, pose_idx, view_idx):
        cam_name = self.cam_names[view_idx]

        color_img = cv.imread(self.data_dir + '/4x/rgbs/%s/%s_rgb%06d.jpg' % (cam_name, cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(self.data_dir + '/4x/masks/%s/%s_mask%06d.png' % (cam_name, cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        return color_img, mask_img

    def load_cam_data(self):
        import csv
        cam_names = []
        ty = []
        extr_mats = []
        intr_mats = [] 
        img_widths = []
        img_heights = []
        with open(self.data_dir + '/4x/calibration.csv', "r", newline = "", encoding = 'utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                cam_names.append(row['name'])
                img_widths.append(int(row['w']))
                img_heights.append(int(row['h']))
                extr_mat = np.identity(4, np.float32)
                extr_mat[:3, :3] = cv.Rodrigues(np.array([float(row['rx']), float(row['ry']), float(row['rz'])], np.float32))[0]
                extr_mat[:3, 3] = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])

                extr_mats.append(extr_mat)  # c2w

                intr_mat = np.identity(3, np.float32)
                intr_mat[0, 0] = float(row['fx']) * float(row['w'])
                intr_mat[0, 2] = float(row['px']) * float(row['w'])
                intr_mat[1, 1] = float(row['fy']) * float(row['h'])
                intr_mat[1, 2] = float(row['py']) * float(row['h'])
                intr_mats.append(intr_mat)

                ty.append(row['ty'])

        self.cam_names, self.img_widths, self.img_heights, self.extr_mats, self.intr_mats ,self.ty\
            = cam_names, img_widths, img_heights, extr_mats, intr_mats,ty

    def getNerfppNorm(self):
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for cam in self.extr_mats:
            # W2C = getWorld2View2(cam.R, cam.T)
            # C2W = np.linalg.inv(cam)
            C2W = cam
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        self.radius =radius
        return {"translate": translate, "radius": radius}

class DatasetDNARendering(DatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        test_cam_ids = None,
    ):
        super(DatasetDNARendering, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            test_cam_ids,
        )

    def load_color_mask_images(self, pose_idx, view_idx):
        # cam_names is cam_label
        cam_names = self.cam_names[view_idx]

        color_img = cv.imread(self.data_dir + '/images/%s/%06d.jpg' % (cam_names, pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(self.data_dir + '/fmasks/%s/%06d.png' % (cam_names, pose_idx), cv.IMREAD_UNCHANGED)
        return color_img, mask_img

    def load_cam_data(self):
        # 打开并加载 JSON 数据
        json_path = self.data_dir +'/transforms.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 初始化用于存储相机数据的列表
        cam_names = []
        img_widths = []
        img_heights = []
        extr_mats = []
        intr_mats = []

        # 遍历 JSON 文件中的每一个相机 "frame"
        for frame in data['frames']:
            # 1. 提取基本信息
            cam_names.append(frame['camera_label'])
            img_widths.append(frame['w'])
            img_heights.append(frame['h'])

            # 2. 构建内参矩阵 (Intrinsic Matrix)
            # JSON 中的焦距和主点已经是像素单位，可以直接使用
            intr_mat = np.identity(3, dtype=np.float32)
            intr_mat[0, 0] = frame['fl_x']
            intr_mat[1, 1] = frame['fl_y']
            intr_mat[0, 2] = frame['cx']
            intr_mat[1, 2] = frame['cy']
            intr_mats.append(intr_mat)

            # k1 = frame['k1']
            # k2 = frame['k2']
            # p1 = frame['p1']
            # p2 = frame['p2']

            c2w_opengl_extr_mat = np.array(frame['transform_matrix'], dtype=np.float32)

            c2w_opengl_extr_mat[:3, 1:3] *= -1

            c2w_opencv_extr_mat = c2w_opengl_extr_mat

            # trans_opencv_to_opengl = np.array([
            #     [1,  0,  0,  0],
            #     [0, -1,  0,  0],
            #     [0,  0, -1,  0],
            #     [0,  0,  0,  1]
            # ], dtype=np.float32)

            # c2w_opencv_extr_mat = trans_opencv_to_opengl @ c2w_opengl_extr_mat @ np.linalg.inv(trans_opencv_to_opengl)

            # extr_mat = c2w_opengl_extr_mat @ trans_opencv_to_opengl

            extr_mats.append(c2w_opencv_extr_mat)
        
        self.cam_names, self.img_widths, self.img_heights, self.extr_mats, self.intr_mats \
            = cam_names, img_widths, img_heights, extr_mats, intr_mats

    def getNerfppNorm(self):
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for cam in self.extr_mats:
            # W2C = getWorld2View2(cam.R, cam.T)
            # C2W = np.linalg.inv(cam)
            C2W = cam
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        self.radius =radius
        return {"translate": translate, "radius": radius}