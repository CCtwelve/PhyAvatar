from __future__ import annotations

import os
import os.path as osp
import subprocess
import importlib.util
from collections.abc import Sequence

import fire
import torch

import sys

PROJECT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from utils.gs_generation.preprocess.utils.process_utils import normalize_process_subdirs
except ModuleNotFoundError:
    try:
        from preprocess.utils.process_utils import normalize_process_subdirs
    except ModuleNotFoundError:
        _process_utils_path = osp.join(osp.dirname(__file__), "utils", "process_utils.py")
        _process_utils_spec = importlib.util.spec_from_file_location(
            "process_utils_fallback", _process_utils_path
        )
        if _process_utils_spec is None or _process_utils_spec.loader is None:
            raise
        _process_utils_module = importlib.util.module_from_spec(_process_utils_spec)
        _process_utils_spec.loader.exec_module(_process_utils_module)
        normalize_process_subdirs = _process_utils_module.normalize_process_subdirs

# ckpt_root = os.environ["SAPIENS_CHECKPOINT_ROOT"]


def predict_keypoints(
    images_dir: str,
    out_kp2d_dir: str,
    fmasks_dir: str | None = None,
    sapiens_ckpt_path: str = "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2",
    detector_ckpt_path: str = "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    # sapiens_ckpt_path: str = f"{ckpt_root}/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2",
    # detector_ckpt_path: str = f"{ckpt_root}/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    gpu_ids: Sequence[int] | None = None,
    num_workers: int = 4,
    process_subdirs: Sequence[str] | str | None = None,
):
    if gpu_ids is None:
        gpu_ids = tuple(range(torch.cuda.device_count()))

    normalized_subdirs = normalize_process_subdirs(process_subdirs)
    process_subdirs_arg = None
    if normalized_subdirs:
        process_subdirs_arg = ",".join(normalized_subdirs)

    command = [
        "python",
        "sapiens/lite/demo/vis_pose.py",
        sapiens_ckpt_path,
        "--det-checkpoint",
        detector_ckpt_path,
        "--det-config",
        "sapiens/lite/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py",
        "--images_dir",
        images_dir,
        "--fmasks_dir",
        fmasks_dir if fmasks_dir is not None else "None",
        "--output_dir",
        out_kp2d_dir,
        "--skip_exists",
        "--gpu_ids",
        ",".join(map(str, gpu_ids)),
        "--num_workers",
        str(num_workers),
    ]

    if process_subdirs_arg is not None:
        command.extend(["--process_subdirs", process_subdirs_arg])

    subprocess.run(
        command,
        cwd=os.path.dirname(__file__),
        check=False,
    )


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/predict_keypoints.py --images_dir $DATADIR/images --fmasks_dir $DATADIR/fmasks --out_kp2d_dir $DATADIR/poses_2d
    fire.Fire(predict_keypoints)
