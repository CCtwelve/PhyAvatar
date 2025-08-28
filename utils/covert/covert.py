import numpy as np
import torch
from scipy.spatial.transform import Rotation
def convert(world2cam_ref,gaussian_positions,scale,rotation):

    world2cam_posed = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


    posed2ref_matrix = np.dot(world2cam_ref, np.linalg.inv(world2cam_posed))
    posed2ref_rotation = posed2ref_matrix[:3, :3]
    transl = posed2ref_matrix[:3, 3]

    transformed_positions = (gaussian_positions @ posed2ref_rotation.T + transl)

    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

    rotation_matrices = quaternion_to_matrix(torch.from_numpy(rotation))
    transformed_rotation_matrices = torch.matmul(torch.from_numpy(posed2ref_rotation).transpose(0, 1), rotation_matrices)
    transformed_quaternions = matrix_to_quaternion(transformed_rotation_matrices).numpy()

    # transformed_scale = scale @ transformed_rotation_matrices.T

    return transformed_positions ,transformed_quaternions

