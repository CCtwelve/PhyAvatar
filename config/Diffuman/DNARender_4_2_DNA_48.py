import os
from os.path import join

"""



DNARender_4_2_DNA_48.py

针对 DNARender 数据集的配置文件。

与 ActorsHQ/HQ 配置的主要区别：
- 数据直接来自 Diffuman4D 提供的 DNARender 数据集（diffuman4d_data_root / subject）
- run_seg_clothes_DNA.sh 会先根据 frame_range 从该数据集中复制所需数据到
  PhyAvatar 的 results/nerfstudio/filename（即 soft_path），然后复用
  ns_export / data_preparation / ns_train_clothes 三个动作。
"""

# 为了避免写死绝对路径，这里统一从当前文件位置推导工程根目录
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
code_root = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../PhyAvatar

# ============================================================================
# Base Configuration - Modify these only
# ============================================================================
# 相对于工程根目录的路径
datasets_root = join(code_root, "..", "..", "..", "dataset")
inference_result_root = join(code_root, "..", "Diffuman4D")

# DNARender 数据标识
subject = "0013_01"
frame_range = "000077"  # 仅复制这一帧对应的图片 / json 等

# 为了兼容现有 pipeline，这里仍然提供 sequence / resolution
sequence = "Sequence1"
resolution = "4x"

# 用于标识当前场景 / 结果目录名
filename = "DNARender_0013_01_4_2_48_70"

# Diffuman4D 推理相关
inference_exp = "demo_3d"
inference_script = "inference.py"

# DNARender 源数据根目录（相对 code_root 推导）
diffuman4d_data_root = join(inference_result_root, "data", "datasets--krahets--diffuman4d_example")


# Auto-built paths (don't modify!!!)
colmap_path = join(code_root, "results", "colmap")
nerfstudio_path = join(code_root, "results", "nerfstudio", filename)
script_dir = join(code_root, "utils", "gs_generation")

# 对齐骨架等仍然直接使用 DNARender 源数据中的 subject 目录
align_pose_target_dir = join(diffuman4d_data_root, subject)
comparison_skeletons_dir = join(align_pose_target_dir, "skeletons")
soft_link_path = join(code_root, "results", "nerfstudio")

# 对于 DNARender，我们仍然保留 inference_result_path 的定义以兼容其它脚本，
# 但 run_seg_clothes_DNA.sh 会直接使用 inference_soft_link_path/filename 来构造 soft_path。
inference_result_path = join(inference_result_root, "output", "results", inference_exp, filename)

# DNARender 相关数据现在直接复制到 results/gs_generation 下（由 run_gen_human_DNA.sh 负责复制），
# 这样与普通 HQ/ActorsHQ pipeline 的数据目录保持一致。
# 注意：这里使用的是相对工程根目录的路径，避免写死绝对路径
inference_soft_link_path = join(code_root, "results", "gs_generation")
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


# 对于 DNARender，训练输出默认在 outputs/ 目录下（nerfstudio 默认行为）
# 优先从 outputs/ 目录读取，如果不存在则回退到 results/nerf_result/
_ns_train_base_dir_outputs = join(code_root, "outputs", filename, "splatfacto")
_ns_train_latest_outputs = _latest_subdir(_ns_train_base_dir_outputs)
_ns_train_base_dir_fallback = join(ns_train_output_path, filename, "splatfacto")
_ns_train_latest_fallback = _latest_subdir(_ns_train_base_dir_fallback)

# 优先使用 outputs/ 目录下的训练结果
if _ns_train_latest_outputs:
    _ns_train_base_dir = _ns_train_base_dir_outputs
    _ns_train_latest = _ns_train_latest_outputs
elif _ns_train_latest_fallback:
    _ns_train_base_dir = _ns_train_base_dir_fallback
    _ns_train_latest = _ns_train_latest_fallback
else:
    # 如果都不存在，使用 outputs/ 作为默认路径（即使目录不存在，也会在训练时创建）
    _ns_train_base_dir = _ns_train_base_dir_outputs
    _ns_train_latest = _ns_train_latest_outputs if _ns_train_latest_outputs else "latest"

ns_train_yaml_path = join(_ns_train_base_dir, _ns_train_latest, "config.yml")
ns_export_gspath = join(_ns_train_base_dir, _ns_train_latest)

# seg_clothes parameters
soft_path = join(inference_soft_link_path, filename)
images_dir = join(soft_path, "images_alpha")
masks_dir = join(soft_path, "fmasks")
transforms_json = join(soft_path, "transforms.json")
data_prep_output_root = join(soft_path, "seg_clothes", subject, sequence)
smplx_root = join(datasets_root, "body_models")
actorshq_smplx_zip = join(datasets_root, subject, sequence, "ActorsHQ_smplx.zip")
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
        "datasets_root": datasets_root,
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
        # 对 DNARender，我们希望 nerfstudio 直接读取由 run_seg_clothes_DNA.sh
        # 复制到 PhyAvatar 工程下的 results/gs_generation/filename 目录中的数据，
        # 而不是原始的 Diffuman4D/output/results/... 目录。
        "ns_train": {
            # 原来是 inference_result_path (= Diffuman4D/output/results/demo_3d/filename)
            # 这里改为 soft_path (= code_root/results/gs_generation/filename)
            "data_dir": soft_path,
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

