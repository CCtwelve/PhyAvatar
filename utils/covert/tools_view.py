#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/29 7:52
# @Author  : jc Han
# @help    :
#!/usr/bin/env python3
import numpy as np
from utils.covert.camera_data import CameraData, read_calibration_csv
from scipy.spatial.transform import Rotation
from utils.covert.covert import convert
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from utils.other_util import common_world
from plyfile import PlyData, PlyElement
import numpy as np
import cv2 as  cv
import torch
from utils.general_util import load_gs_attributes,intrinsic_to_fov
def save_gs_ply(gs_dict, output_path):

    # 准备顶点数据
    vertices = np.zeros(gs_dict["positions"].shape[0],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('opacity', 'f4'),
                               ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                               ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                               ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')])

    vertices['x'] = gs_dict["positions"][:, 0]
    vertices['y'] = gs_dict["positions"][:, 1]
    vertices['z'] = gs_dict["positions"][:, 2]
    vertices['opacity'] = gs_dict["opacities"].flatten()

    vertices['f_dc_0'] = gs_dict["features_dc"][:, 0, 0]
    vertices['f_dc_1'] = gs_dict["features_dc"][:, 1, 0]
    vertices['f_dc_2'] = gs_dict["features_dc"][:, 2, 0]

    vertices['scale_0'] = gs_dict["scales"][:, 0]
    vertices['scale_1'] = gs_dict["scales"][:, 1]
    vertices['scale_2'] = gs_dict["scales"][:, 2]

    vertices['rot_0'] = gs_dict["rots"][:, 0]
    vertices['rot_1'] = gs_dict["rots"][:, 1]
    vertices['rot_2'] = gs_dict["rots"][:, 2]
    vertices['rot_3'] = gs_dict["rots"][:, 3]

    vertex_element = PlyElement.describe(vertices, 'vertex')

    PlyData([vertex_element], text=False).write(output_path)
    print(f"save to  {output_path}")




def cover_posed_to_ref(posed_gs_path,ref_gs_path,image_name="Cam127"):

    csv_path = r"/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/calibration.csv"

    cameras = read_calibration_csv(csv_path)
    for camera in cameras:
        if camera.name == image_name:

            extr_mat = np.identity(4, np.double)

            rotation = camera.rotation_axisangle
            trans = camera.translation
            extr_mat[:3, :3] = cv.Rodrigues(np.array([rotation[0], rotation[1], rotation[2]]))[0]
            extr_mat[:3, 3] = np.array([trans[0], trans[1], trans[2]])


            posted_gs= load_gs_attributes(posed_gs_path)

            ref_gs = posted_gs

            transformed_positions ,new_scales = \
                convert(extr_mat, posted_gs["positions"],posted_gs['scales'])

            ref_gs["positions"] = transformed_positions
            ref_gs["scales"] = new_scales

            save_gs_ply(ref_gs,ref_gs_path)


            return  ref_gs


def visualize(csv_path,is_convert=False):
    cameras = read_calibration_csv(csv_path)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.0, 4.0)
    ax.set_zlim(-2.5, 2.5)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    def draw_axis(c,v,color,lw=1,name=None):
        p = c+v
        ax.plot([c[0], p[0]], [c[1], p[1]], [c[2], p[2]], color=color, lw=lw)

        if name!=None:
            midpoint = [(c[0] + p[0]) / 2, (c[1] + p[1]) / 2, (c[2] + p[2]) / 2]
            ax.text(*midpoint, f"{name}", fontsize=5, color='red')

    s = 0.2

    idx = 1
    for camera in cameras:
        c2w_rotation = Rotation.from_rotvec(camera.rotation_axisangle)
        c2w = np.eye(4)
        c2w [:3, :3] = c2w_rotation.as_matrix()
        c2w [:3, 3] = camera.translation


        if camera.name == 'Cam127':

            w2c_Rotation = Rotation.from_rotvec(-camera.rotation_axisangle)
            w2c_ref = np.eye(4)
            w2c_ref[:3, :3] =  w2c_Rotation.as_matrix()

            tvec = -w2c_Rotation.as_matrix() @ camera.translation
            tx, ty, tz = tuple(tvec)
            w2c_ref[:3, 3] = [tx, ty, tz]

            w2c_ref = torch.inverse(torch.tensor(w2c_ref))

            # print(w2c_ref)

            # w2c_ref = np.identity(4, np.double)
            # rotation = camera.rotation_axisangle
            # trans = camera.translation
            # w2c_ref[:3, :3] = cv.Rodrigues(np.array([rotation[0], rotation[1], rotation[2]] ))[0]
            # w2c_ref[:3, 3] = np.array([trans[0], trans[1], trans[2]])

            # posted_gs = load_gs_attributes(
            #     r"/mnt/cvda/cvda_phava/code/Han/LHM/gs_result/2025-08-20 19:31:08/0_0_1755689470.ply")

            posted_gs = load_gs_attributes(
                r"/mnt/cvda/cvda_phava/code/Han/LHM/gs_result/to_covert/0_0_1756293757.ply")

            transformed_positions, transformed_rotation= \
                    convert(w2c_ref, posted_gs["positions"],posted_gs["scales"], posted_gs["rot"])

            # Cam127 cam_posed
            draw_axis(camera.translation, c2w_rotation.apply(np.array([0.4, 0, 0])), "black")
            draw_axis(camera.translation, c2w_rotation.apply(np.array([0, 0.4, 0])), "green")
            draw_axis(camera.translation, c2w_rotation.apply(np.array([0, 0, 0.4])), "red", lw=3,name = camera.name)
            if is_convert :
                # ax.scatter(transformed_positions2 [:, 0], transformed_positions2 [:, 1], transformed_positions2 [:, 2],
                #            c='r', s=0.2, alpha=0.2)

                ax.scatter(transformed_positions[:, 0], transformed_positions[:, 1]  , transformed_positions[:, 2],
                           c='g', s=0.2, alpha=0.2)

            ref_gs = posted_gs

            ref_gs["positions"] = transformed_positions
            ref_gs['rots'] = transformed_rotation

            save_gs_ply(ref_gs,'/mnt/cvda/cvda_phava/code/Han/PhyAvatar/utils/covert/result.ply')

        else:
            draw_axis(camera.translation, c2w_rotation.apply(np.array([s, 0, 0])), "red")
            draw_axis(camera.translation, c2w_rotation.apply(np.array([0, s, 0])), "green")
            draw_axis(camera.translation, c2w_rotation.apply(np.array([0, 0, s])), "blue")

        idx = idx+1
    # cam_ref
    # draw_axis([0, 0, 0], c2w_rotation.apply(np.array([0.4, 0, 0])), "black")
    # draw_axis([0, 0, 0], c2w_rotation.apply(np.array([0, 0.4, 0])), "green")
    # draw_axis([0, 0, 0], c2w_rotation.apply(np.array([0, 0, 0.4])), "red", lw=4,name="posed")
    ax.view_init(elev=40, azim=-90)
    # ax.view_init(elev=60, azim=-90)
    # plt.show()

    # plt.savefig('/mnt/cvda/cvda_phava/code/Han/PhyAvatar/utils/covert/result.png')

def crop_image(gt_mask, patch_size, randomly,bg, *args):
    """
    :param gt_mask: (H, W)
    :param patch_size: resize the cropped patch to the given patch_size
    :param randomly: whether to randomly sample the patch
    :param args: input images with shape of (C, H, W)
    """
    mask_uv = torch.argwhere(gt_mask > 0.)
    min_v, min_u = mask_uv.min(0)[0]
    max_v, max_u = mask_uv.max(0)[0]
    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    if randomly and max_size > patch_size:
        random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
    for image in args:
        cropped_image = bg[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
        if len_v > len_u:
            start_u = (max_size - len_u) // 2
            cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
        else:
            start_v = (max_size - len_v) // 2
            cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

        if randomly and max_size > patch_size:
            cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
        else:
            cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
        cropped_images.append(cropped_image)



    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]

if __name__ == "__main__":

    csv_path = r"/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/calibration.csv"

    visualize(csv_path,is_convert=True)


