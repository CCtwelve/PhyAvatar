#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/17 16:18
# @Author  : jc Han
# @help    :
import trimesh
import json
import numpy as np
import torch
import smplx
import config
if __name__ == "__main__":
    # source images
    model = 'animatable'

    # smplx_obj = "/mnt/cvda/cvda_phava/code/Han/LHM/train_data/custom_motion/custom_dress/smplx_params/00001.json"
    smplx_obj = '/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1'
    smplx_dir = '/mnt/cvda/cvda_phava/code/Han/PhyAvatar/utils/smplx_model'

    if model == 'LHM':
        with open(smplx_obj) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }


    else:
        smpl_model = smplx.SMPLX(model_path=smplx_dir , gender='neutral', use_pca=False,
                                 num_pca_comps=45, flat_hand_mean=True, batch_size=1)

        smplx_params = np.load('/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/Actor01/Sequence1/smpl_params.npz')
        smplx_params = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smplx_params.items()}

    for key in smplx_params.keys():
        param = smplx_params[key]
        print(f"  {key}: shape={param.shape}, dtype={param.dtype}")

    total_frames = smplx_params['global_orient'].shape[0] - 1500
    print(total_frames)
    for idx in range( total_frames ):

        print(f"处理第 {idx}/{total_frames+48} 帧...")

        frame_params = {}
        for key in smplx_params.keys():
            if key == 'betas':
                frame_params[key] = smplx_params[key][0:1]
            else:
                frame_params[key] = smplx_params[key][idx:idx + 1]

        HQ_smpl = smpl_model.forward(betas = frame_params['betas'],
                                       global_orient =frame_params['global_orient'],
                                       transl =frame_params['transl'],
                                       body_pose = frame_params['body_pose'],
                                         jaw_pose = frame_params['jaw_pose'],
                                         expression= frame_params['expression'],
                                         left_hand_pose =frame_params['left_hand_pose'],
                                         right_hand_pose = frame_params['right_hand_pose']
                                        )

        json_forward_params = {
            'betas': frame_params['betas'][0].cpu().numpy().tolist(),
            'global_orient': frame_params['global_orient'].cpu().numpy().tolist(),
            'transl': frame_params['transl'].cpu().numpy().tolist(),
            'body_pose': frame_params['body_pose'].reshape(-1,3).cpu().numpy().tolist(),
            'jaw_pose': frame_params['jaw_pose'].cpu().numpy().tolist(),
            'expression': frame_params['expression'].cpu().numpy().tolist(),
            'left_hand_pose': frame_params['left_hand_pose'].reshape(-1,3).cpu().numpy().tolist(),
            'right_hand_pose': frame_params['right_hand_pose'].reshape(-1,3).cpu().numpy().tolist()
        }

        with open(f'/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/smplx_model_json/{48+idx}.json', 'w') as f:
            json.dump(json_forward_params, f, indent=2)

        # 获取顶点和面片
        vertices = HQ_smpl.vertices.detach().cpu().numpy()[0]  # [10475, 3]
        faces = smpl_model.faces

        # 创建trimesh对象并保存为PLY
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(f'/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/smplx_model/{48+idx}.ply')

        print("✅ PLY文件已保存: smplx_model.ply")