from os.path import join

# ============================================================================
# Base Configuration - Modify these only
# ============================================================================
HQ_data_root = "/mnt/cvda/cvda_phava/dataset"
code_root = "/mnt/cvda/cvda_phava/code/Han/PhyAvatar"
inference_result_root = "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results"
subject = "Actor01"
sequence = "Sequence1"
resolution = "4x"
filename = "nerfstudio_4_2_48"
example_dataset = "0013_01"
inference_exp = "demo_3d"
inference_script = "inference.py"
diffuman4d_data_root = "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/data/datasets--krahets--diffuman4d_example"

# Auto-built paths
data_path = join(HQ_data_root, subject, sequence, resolution)
colmap_path = join(data_path, "colmap")
nerfstudio_path = join(inference_result_root,inference_exp, filename)
script_dir = join(code_root, "utils", "gs_generation")
align_pose_target_dir = join(diffuman4d_data_root, example_dataset)
comparison_skeletons_dir = join(align_pose_target_dir, "skeletons")

# ============================================================================
# Configuration Dictionary
# ============================================================================
config = {
    "pre_view": [127, 40, 95, 152], # inference view
    "front_view": [127, 40, 95, 152, 5, 6, 7, 8, 21, 22, 23, 24, 109, 110, 111, 112, 126, 128, 135, 136, 143, 144],
    "side_view": [37, 38, 39, 93, 94, 96],
    "back_view": [53, 54, 55, 56, 77, 78, 79, 80, 151, 159, 160],
    "args_config": {
        "HQ_data_root": HQ_data_root,
        "subject": subject,
        "sequence": sequence,
        "resolution": resolution,
        "triangulator": True,
        "ff": 460,
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
        "default_predict_keypoints_view": "09,04,05,17",
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
            "data_dir": data_path,
            "inference_script_path": inference_script,
        },
        "visualize_cameras": {
            "transforms_name": "transforms",
            "scale": 0.2,
        },
        "ns_train": {
            "data_dir": nerfstudio_path ,
        },
    },
}

