#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/3 22:54
# @Author  : jc Han
# @help    :
import glob
import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from utils.graphics_util import getWorld2View2
from scipy.spatial.transform import Rotation
class DatasetBase(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
    ):
        super(Dataset, self).__init__()
        print(data_dir,frame_range)
        self.data_dir = data_dir
        self.training = training

        print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
        frame_range = range(frame_range[0], frame_range[1], frame_range[2])
        self.pose_list = list(frame_range)

        if self.training:
            if used_cam_ids is None:
                self.used_cam_ids = list(range(self.view_num))
            else:
                self.used_cam_ids = used_cam_ids
            print('# Used camera ids: ', self.used_cam_ids)

            self.data_list = []
            for pose_idx in self.pose_list:
                for view_idx in self.used_cam_ids:
                    self.data_list.append((pose_idx, view_idx - 1 ))

        self.load_cam_data()

        self.getNerfppNorm()

    def __len__(self):
        if self.training:
            return len(self.data_list)
        else:
            return len(self.pose_list)

    def __getitem__(self, index):
        return self.getitem(index, self.training)

    def getitem(self, index):

        pose_idx, view_idx = self.data_list[index]
        data_idx = (pose_idx, view_idx)
        # print('data index: (%d, %d)' % (pose_idx, view_idx))

        data_item = dict()
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx

        color_img, mask_img = self.load_color_mask_images(pose_idx, view_idx)

        color_img = (color_img / 255.).astype(np.float32)

        boundary_mask_img, mask_img = self.get_boundary_mask(mask_img)
        # print(self.radius)
        data_item.update({
            'img_name': self.cam_names[view_idx],
            'img_h': color_img.shape[0],
            'img_w': color_img.shape[1],
            'extr': self.extr_mats[view_idx],
            'intr': self.intr_mats[view_idx],
            'color_img': color_img,
            'mask_img': mask_img,
            'boundary_mask_img': boundary_mask_img,
            'ty':self.ty[view_idx],
            'radius':self.radius,
        })

        return data_item

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
        training = True
    ):
        super(DatasetActorsHQ, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training
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