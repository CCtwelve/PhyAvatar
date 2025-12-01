import os
from os.path import join

# ============================================================================
# Base Configuration - Modify these only
# ============================================================================
HQ_data_root = "/mnt/cvda/cvda_phava/dataset/Chunxun_local/ActorsHQ/tmp/actorshq"
code_root = "/mnt/cvda/cvda_phava/code/Han/PhyAvatar"
inference_result_root = "/mnt/cvda/cvda_phava/code/Han/Diffuman4D"
subject = "Actor02"
sequence = "Sequence1"
resolution = "4x"
filename = "HQ_Actor02_4_2_48_70"
inference_exp = "demo_3d"
inference_script = "inference.py"
diffuman4d_data_root = "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/data/datasets--krahets--diffuman4d_example"


# Auto-built paths (don't modify!!!)
# HQ_data_path = join(HQ_data_root, subject, sequence, resolution)
colmap_path = join(code_root, "results", "colmap")
nerfstudio_path = join(code_root, "results", "nerfstudio", filename)
script_dir = join(code_root, "utils", "gs_generation")
align_pose_target_dir = join(diffuman4d_data_root, "0013_01")
comparison_skeletons_dir = join(align_pose_target_dir, "skeletons")
soft_link_path = join(code_root, "results", "nerfstudio")
inference_result_path = join(inference_result_root, "output", "results", inference_exp, filename)
inference_soft_link_path = join(code_root, "results","gs_generation")
ns_train_output_path = join(code_root, "results", "nerf_result")
seg_clothes_path = join(inference_result_path, "seg_clothes")

# seg_clothes paths
def _latest_subdir(parent: str) -> str:
    """Return the lexicographically latest subdirectory name under parent, or '' if none."""
    try:
        entries = [
            d
            for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d))
        ]
    except FileNotFoundError:
        return ""
    if not entries:
        return ""
    entries.sort()
    return entries[-1]


_ns_train_base_dir = join(ns_train_output_path, filename,"splatfacto")
_ns_train_latest =  _latest_subdir(_ns_train_base_dir)
ns_train_yaml_path = join(_ns_train_base_dir, _ns_train_latest,"config.yml")
ns_export_gspath = join(_ns_train_base_dir, _ns_train_latest)

# seg_clothes parameters
soft_path = join(inference_soft_link_path, filename)
images_dir = join(soft_path, "images_alpha")
masks_dir = join(soft_path, "fmasks")
transforms_json = join(soft_path, "transforms.json")
data_prep_output_root = join(soft_path,"seg_clothes", subject, sequence)
smplx_root = join(HQ_data_root, "body_models")
actorshq_smplx_zip = join(HQ_data_root, subject, sequence, "ActorsHQ_smplx.zip")
actorshq_smplx_del_path = join(data_prep_output_root, "ActorsHQ_smplx")
# 专门用于衣服训练的 ns-train 输出路径
ns_train_clothes_output_path = join(ns_train_output_path, filename)


# ============================================================================
# Configuration Dictionary
# ============================================================================
config = {
    "pre_view": [127, 40, 95, 152], # inference view
    "front_view": [127, 40, 95, 152, 5, 6, 7, 8, 21, 22, 23, 24, 109, 110, 111, 112, 126, 128, 135, 136, 143, 144],
    "side_view": [37, 38, 39, 93, 94, 96],
    "back_view": [53, 54, 55, 56, 77, 78, 79, 80, 151, 159, 160],
    "colmap_construction.py": {
        "HQ_data_root": HQ_data_root,
        "subject": subject,
        "sequence": sequence,
        "resolution": resolution,
        "triangulator": True,
        "ff": 70,
        "no_gpu": False,
        "is_ba": True,
        "n": 5,
        "colmap_root": colmap_path,
    },
    "colmap2nerf_args": {
        "root": colmap_path,
        "target": nerfstudio_path,
        "is_ba": "0",
    },
    "run_sh_config": {
        "root_dir": code_root,
        "script_dir": script_dir,
        "default_datadir": nerfstudio_path,
        "default_predict_keypoints_view": "36,11,29,22",
        "default_use_align_pose": True,
        "actions_envs": {
            "colmap_construction": "colmap-cuda118",
            "colmap2nerf": "nerfstudio",
            "remove_background": "diffuman4d",
            "predict_keypoints": "sapiens_lite",
            "triangulate_skeleton": "diffuman4d",
            "draw_skeleton": "diffuman4d",
            "align_pose_pcd": "diffuman4d",
            "project_pcd_skeletons": "diffuman4d",
            "overlay_skeletons": "diffuman4d",
            "collage": "diffuman4d",
            "sync_cameras": "diffuman4d",
            "inference": "diffuman4d",
        },
        "remove_background": {
            "model_name": "ZhengPeng7/BiRefNet",
            "batch_size": 8,
        },
        "align_pose_pcd": {
            "target_dir": align_pose_target_dir,
            "source_frame": "000000",
            "target_frame": "000000",
            "run_icp": False,
        },
        "collage": {
            "comparison_images_dir": comparison_skeletons_dir,
            "images_per_row": 8,
        },
        "inference": {
            "exp": inference_exp,
            "scene_label": filename,
            # 让 Diffuman4D 直接读取 PhyAvatar 的 results 目录
            # camera_path_pat: '{data_dir}/{scene_label}/transforms.json'
            # ⇒ data_dir=soft_link_path=/.../code/Han/PhyAvatar/results
            #    scene_label=filename=nerfstudio_4_2_48_70
            "data_dir": soft_link_path,
            "inference_script_path": inference_script,
        },
        "visualize_cameras": {
            "transforms_name": "transforms",
            "scale": 0.2,
        },
        "ns_train": {
            "data_dir": inference_result_path,
            "output_dir": ns_train_output_path,
        },
        # 专门用于衣服训练的 ns-train 配置（供 run_seg_clothes.sh 使用）
        "ns_train_clothes": {
            # 数据目录使用 seg_clothes_path（如 /.../results/seg_clothes/nerfstudio_4_2_48_70_seg_clothes）
            "data_dir": seg_clothes_path,
            # 输出目录使用 ns_train_clothes_output_path（如 /.../results/nerf_result/nerfstudio_4_2_48_70）
            "output_dir": ns_train_clothes_output_path,
        },
    },
    "data_preparation.py": {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "transforms_json": transforms_json,
        "output_root": data_prep_output_root,
        "smplx_root": smplx_root,
        "actorshq_smplx_zip": actorshq_smplx_zip,
        "actorshq_smplx_del_path": actorshq_smplx_del_path,
        "masker_prompt": "shirt, shorts",
        "gender": "male",
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "skip_masking": False,
        "skip_reorg": False,
        "copy_data": False,
        "skip_smplx": True,
        "skip_json": False,
        "skip_cloth_extraction": False,
    },
}

