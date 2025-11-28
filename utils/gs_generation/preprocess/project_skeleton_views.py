#!/usr/bin/env python3
import os
import os.path as osp
import sys
import json
import importlib.util

import fire
import numpy as np
from easyvolcap.utils.console_utils import tqdm
from easyvolcap.utils.parallel_utils import parallel_execution

PROJECT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_local_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPROCESS_DIR = osp.dirname(osp.abspath(__file__))
UTILS_DIR = osp.join(PREPROCESS_DIR, "utils")

try:
    from src.data.utils.camera_parser import parse_cameras
except ModuleNotFoundError:
    parse_cameras = _load_local_module(
        osp.join(UTILS_DIR, "camera_parser.py"), "camera_parser_local"
    ).parse_cameras

try:
    from utils.gs_generation.preprocess.utils.triang_utils import triangulate_points, project_points
except ModuleNotFoundError:
    triang_module = _load_local_module(
        osp.join(UTILS_DIR, "triang_utils.py"), "triang_utils_local"
    )
    triangulate_points = triang_module.triangulate_points
    project_points = triang_module.project_points


def read_kp2d(path, dtype=np.float64):
    with open(path, "r") as f:
        pose = json.load(f)
    inst = pose["instance_info"][0]
    kp = np.array(inst["keypoints"], dtype=dtype)
    kp_score = np.array(inst.get("keypoint_scores", np.ones(len(kp))), dtype=dtype)
    return kp, kp_score


def write_kp2d(path, kp, kp_depth=None, kp_score=None):
    inst = {"keypoints": kp.tolist()}
    if kp_depth is not None:
        inst["keypoint_depths"] = kp_depth.tolist()
    if kp_score is not None:
        inst["keypoint_scores"] = kp_score.tolist()
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"instance_info": [inst]}, f, indent=4)


def write_kp3d(path, kp3d, kp3d_reproj):
    inst = {"keypoints": kp3d.tolist(), "keypoint_reproj": kp3d_reproj.tolist()}
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"instance_info": [inst]}, f, indent=4)


def project_skeleton_from_views(
    camera_path: str,
    kp2d_dir: str,
    out_kp3d_dir: str,
    out_kp2d_dir: str,
    triang_cam_labels,
    spa_label_range: list[int] | None = None,
    tem_label_range: list[int] | None = None,
    skip_exists: bool = False,
    intri_scale: float | None = None,
    kp2d_padding: list[float] | None = None,
    num_workers: int = 1,
    dtype=np.float64,
    make_kpmap: bool = False,
    out_kpmap_dir: str | None = None,
    kpmap_ext: str = ".webp",
    kp2d_canvas_shape: tuple[int, int] = (1024, 1024),
    kpmap_shape: tuple[int, int] = (1024, 1024),
    kpmap_image_quality: int = 85,
    kpmap_radius: float = 9.0,
    kpmap_thickness: float = 9.0,
    kpmap_draw_face_keypoints: bool = False,
):
    def _normalize_labels(labels):
        if isinstance(labels, str):
            tokens = labels.replace(",", " ").split()
            return [int(t) for t in tokens]
        if isinstance(labels, (list, tuple)):
            return [int(x) for x in labels]
        return [int(labels)]

    def _flag_provided(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv)

    triang_cam_labels = _normalize_labels(triang_cam_labels)

    if len(triang_cam_labels) != 4 and "--triang_cam_labels" in sys.argv:
        idx = sys.argv.index("--triang_cam_labels")
        collected = []
        for token in sys.argv[idx + 1 :]:
            if token.startswith("--"):
                break
            collected.extend(token.replace(",", " ").split())
        triang_cam_labels = [int(tok) for tok in collected if tok]

    if not _flag_provided("--spa_label_range"):
        spa_label_range = None
    if not _flag_provided("--tem_label_range"):
        tem_label_range = None
    if not _flag_provided("--skip_exists"):
        skip_exists = False
    if not _flag_provided("--intri_scale"):
        intri_scale = None
    if not _flag_provided("--kp2d_padding"):
        kp2d_padding = None
    if not _flag_provided("--num_workers"):
        num_workers = 1
    if not _flag_provided("--make_kpmap"):
        make_kpmap = False

    assert len(triang_cam_labels) == 4, "triang_cam_labels 需指定四个视角"

    cams = parse_cameras(camera_path, coord_system="opencv", normalize_scene=False)
    all_labels = sorted(cams.keys())

    if spa_label_range is not None:
        if isinstance(spa_label_range, str):
            spa_label_range = [int(x.strip()) for x in spa_label_range.split(",") if x.strip()]
        elif not isinstance(spa_label_range, (list, tuple)):
            spa_label_range = [int(spa_label_range)]
        if len(spa_label_range) not in (1, 3):
            raise ValueError("spa_label_range 需提供 1 个或 3 个整数")
        if len(spa_label_range) == 1:
            all_labels = [f"{int(spa_label_range[0]):02d}"]
        else:
            b, e, s = spa_label_range
            all_labels = [f"{i:02d}" for i in range(b, e, s)]

    tri_labels = [f"{int(i):02d}" for i in triang_cam_labels]
    proj_labels = [label for label in all_labels if label not in tri_labels]

    sample_dir = osp.join(kp2d_dir, tri_labels[0])
    existing_tem_labels = sorted(f.split(".")[0] for f in os.listdir(sample_dir) if f.endswith(".json"))
    frame_int_to_name = {}
    for name in existing_tem_labels:
        try:
            frame_int_to_name[int(name)] = name
        except ValueError:
            frame_int_to_name[name] = name

    def resolve_tem_label(val):
        if isinstance(val, str):
            if val in existing_tem_labels:
                return val
            val = val.strip()
            if val.isdigit() and int(val) in frame_int_to_name:
                return frame_int_to_name[int(val)]
            return val
        try:
            ival = int(val)
        except Exception:
            return str(val)
        return frame_int_to_name.get(ival, f"{ival:06d}")

    if tem_label_range is not None:
        if isinstance(tem_label_range, str):
            tem_label_range = [int(x.strip()) for x in tem_label_range.split(",") if x.strip()]
        elif not isinstance(tem_label_range, (list, tuple)):
            tem_label_range = [int(tem_label_range)]
        if len(tem_label_range) not in (1, 3):
            raise ValueError("tem_label_range 需提供 1 个或 3 个整数")
        if len(tem_label_range) == 1:
            tem_labels = [resolve_tem_label(tem_label_range[0])]
        else:
            b, e, s = tem_label_range
            tem_labels = [resolve_tem_label(i) for i in range(b, e, s)]
    else:
        tem_labels = existing_tem_labels

    Ks_tri = np.array([cams[label]["K"] for label in tri_labels], dtype=dtype)
    Ts_tri = np.array([np.linalg.inv(cams[label]["pose"]) for label in tri_labels], dtype=dtype)
    Ks_proj = np.array([cams[label]["K"] for label in proj_labels], dtype=dtype)
    Ts_proj = np.array([np.linalg.inv(cams[label]["pose"]) for label in proj_labels], dtype=dtype)

    if intri_scale is not None:
        for Ks in (Ks_tri, Ks_proj):
            if Ks.ndim == 3 and Ks.size > 0:
                Ks *= intri_scale
                Ks[:, -1, -1] = 1.0

    if kp2d_padding is not None:
        kp2d_padding = np.array(kp2d_padding, dtype=dtype)

    kpmap_base_dir = out_kpmap_dir or out_kp2d_dir
    if make_kpmap and kpmap_ext and not kpmap_ext.startswith("."):
        kpmap_ext = f".{kpmap_ext}"

    def _process_frame(tem_label: str):
        tri_paths = [f"{kp2d_dir}/{label}/{tem_label}.json" for label in tri_labels]
        kp2d, kp2d_score = zip(*[read_kp2d(p, dtype=dtype) for p in tri_paths])
        kp2d = np.stack(kp2d)
        kp2d_score = np.stack(kp2d_score)
        if kp2d_padding is not None:
            kp2d += kp2d_padding[None, None]

        kp3d, reproj, _ = triangulate_points(Ks_tri, Ts_tri, kp2d, kp2d_score=kp2d_score)
        write_kp3d(f"{out_kp3d_dir}/{tem_label}.json", kp3d, reproj)

        if len(proj_labels) == 0:
            return
        kp2d_proj, kp2d_depth_proj, _ = project_points(kp3d, Ks_proj, Ts_proj, kp3d_score=None)
        for i, label in enumerate(proj_labels):
            out_path = f"{out_kp2d_dir}/{label}/{tem_label}.json"
            write_kp2d(out_path, kp2d_proj[i], kp_depth=kp2d_depth_proj[i], kp_score=None)

    if num_workers > 1:
        parallel_execution(
            tem_labels,
            action=_process_frame,
            print_progress=True,
            desc="Triangulate + project",
            num_workers=num_workers,
            sequential=False,
        )
    else:
        for tem_label in tqdm(tem_labels, desc="Triangulate + project"):
            _process_frame(tem_label)

    if make_kpmap and len(proj_labels) > 0 and len(tem_labels) > 0:
        from utils.gs_generation.preprocess.draw_skeleton import draw_skeleton as render_kpmap

        def _convert_labels_to_int(labels):
            try:
                return [int(label) for label in labels]
            except ValueError:
                return None

        def _convert_tem_to_int(labels):
            try:
                return [int(label) for label in labels]
            except ValueError:
                return labels

    render_kpmap(
            kp2d_dir=out_kp2d_dir,
            out_kpmap_dir=kpmap_base_dir,
            kp2d_canvas_shape=kp2d_canvas_shape,
            out_kpmap_shape=kpmap_shape,
            spa_labels=_convert_labels_to_int(proj_labels),
            tem_labels=_convert_tem_to_int(tem_labels),
            image_ext=kpmap_ext,
            image_quality=kpmap_image_quality,
            num_workers=num_workers,
            skip_exists=skip_exists,
            radius=kpmap_radius,
            thickness=kpmap_thickness,
            draw_face_keypoints=kpmap_draw_face_keypoints,
        )


if __name__ == "__main__":
    fire.Fire(project_skeleton_from_views)