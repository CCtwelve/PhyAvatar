#!/usr/bin/env bash
set -euo pipefail

# DNARender 专用的「生成/渲染」脚本：
# - 负责从 Diffuman4D DNARender 源数据复制一份到 PhyAvatar 的 results/nerfstudio/<filename>
# - 然后调用 Diffuman4D 的 inference.py 进行渲染

ROOT_DIR_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${ROOT_DIR_FALLBACK}/config/Diffuman/DNARender_4_2_DNA_48.py"

# 允许通过 --config 覆盖配置文件
if [[ "${1-}" == "--config" && $# -ge 2 ]]; then
  CONFIG_FILE=$2
  shift 2
fi

# 从 Python 配置里取出路径和关键参数
eval "$(
python - "$CONFIG_FILE" <<'PY'
import importlib.util, sys, os

config_path = sys.argv[1]

def emit(name, value):
    if value is None:
        return
    value = str(value).replace("'", "'\"'\"'")
    print(f"{name}='{value}'")

spec = importlib.util.spec_from_file_location("cfg", config_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 来自 DNARender 配置的基础信息
emit("DATA_PREP_SUBJECT", getattr(mod, "subject", None))
emit("FRAME_RANGE", getattr(mod, "frame_range", None))
emit("DIFFUMAN4D_DATA_ROOT", getattr(mod, "diffuman4d_data_root", None))
emit("INFERENCE_SOFT_LINK_PATH", getattr(mod, "inference_soft_link_path", None))
emit("SCENE_FILENAME", getattr(mod, "filename", None))
emit("INFERENCE_RESULT_ROOT", getattr(mod, "inference_result_root", None))

cfg = getattr(mod, "config", {})
run_cfg = cfg.get("run_sh_config", {})

# inference 段：供 Diffuman4D 推理使用
inf_cfg = run_cfg.get("inference", {})
emit("INFERENCE_EXP", inf_cfg.get("exp"))
emit("INFERENCE_SCENE_LABEL", inf_cfg.get("scene_label"))
emit("INFERENCE_DATA_DIR", inf_cfg.get("data_dir"))
emit("INFERENCE_SCRIPT_PATH", inf_cfg.get("inference_script_path"))
PY
)"

if [[ -z "${DATA_PREP_SUBJECT:-}" || -z "${DIFFUMAN4D_DATA_ROOT:-}" || -z "${SCENE_FILENAME:-}" ]]; then
  echo ">> Error: DNARender 配置缺少 subject / diffuman4d_data_root / filename"
  exit 1
fi

if [[ -z "${FRAME_RANGE:-}" ]]; then
  echo ">> Error: DNARender 配置缺少 frame_range"
  exit 1
fi

copy_dnarender_data() {
  # DNARender 数据复制：从 Diffuman4D 数据集拷贝到 PhyAvatar results/nerfstudio
  local SOURCE_ROOT="${DIFFUMAN4D_DATA_ROOT}/${DATA_PREP_SUBJECT}"
  local TARGET_ROOT="${INFERENCE_SOFT_LINK_PATH}/${SCENE_FILENAME}"

  echo ">> DNARender 数据复制开始"
  echo "   源目录:      ${SOURCE_ROOT}"
  echo "   目标目录:    ${TARGET_ROOT}"
  echo "   帧范围(frame_range): ${FRAME_RANGE}"

  if [[ ! -d "${SOURCE_ROOT}" ]]; then
    echo ">> Error: 源目录不存在: ${SOURCE_ROOT}"
    exit 1
  fi

  mkdir -p "${TARGET_ROOT}"

  # 完整复制的内容：grids / sparse_pcd.ply / transforms_input.json / transforms.json / poses_pcd
  if [[ -d "${SOURCE_ROOT}/grids" ]]; then
    echo ">> 复制 grids 目录"
    rm -rf "${TARGET_ROOT}/grids"
    cp -r "${SOURCE_ROOT}/grids" "${TARGET_ROOT}/"
  fi

  # poses_pcd 整个目录完整复制（保留所有帧的点云）
  if [[ -d "${SOURCE_ROOT}/poses_pcd" ]]; then
    echo ">> 复制 poses_pcd 目录（完整）"
    rm -rf "${TARGET_ROOT}/poses_pcd"
    cp -r "${SOURCE_ROOT}/poses_pcd" "${TARGET_ROOT}/"
  fi

  # 将 poses_pcd/<frame_range>.ply 复制为目标目录下的 sparse_pcd.ply
  local FRAME_PCD_SRC="${SOURCE_ROOT}/poses_pcd/${FRAME_RANGE}.ply"
  if [[ -f "${FRAME_PCD_SRC}" ]]; then
    echo ">> 使用 poses_pcd/${FRAME_RANGE}.ply 作为 sparse_pcd.ply"
    cp "${FRAME_PCD_SRC}" "${TARGET_ROOT}/sparse_pcd.ply"
  elif [[ -f "${SOURCE_ROOT}/sparse_pcd.ply" ]]; then
    echo ">> 未找到 poses_pcd/${FRAME_RANGE}.ply，退回复制原始 sparse_pcd.ply"
    cp "${SOURCE_ROOT}/sparse_pcd.ply" "${TARGET_ROOT}/"
  fi

  for tf in transforms_input.json transforms.json; do
    if [[ -f "${SOURCE_ROOT}/${tf}" ]]; then
      echo ">> 复制 ${tf}"
      cp "${SOURCE_ROOT}/${tf}" "${TARGET_ROOT}/"
    fi
  done

  # 根据配置中的 frame_range 重写 transforms.json 里的 file_path 帧号
  if [[ -n "${FRAME_RANGE:-}" && -f "${TARGET_ROOT}/transforms.json" ]]; then
    echo ">> 使用 frame_range=${FRAME_RANGE} 重写 transforms.json 中的 file_path 帧号"
    python - "${TARGET_ROOT}/transforms.json" "${FRAME_RANGE}" <<'PY'
import json, os, sys

transforms_path = sys.argv[1]
frame_range = sys.argv[2]

with open(transforms_path, "r") as f:
    data = json.load(f)

frames = data.get("frames", [])
base_dir = os.path.dirname(transforms_path)

for fr in frames:
    fp = fr.get("file_path")
    if not isinstance(fp, str):
        continue

    head, tail = os.path.split(fp)
    if not tail:
        fr["file_path"] = fp
        continue

    base, ext = os.path.splitext(tail)
    # 仅当文件名本身是数字帧号时才替换，例如 000000.xxx
    if base.isdigit():
        # 自动根据磁盘上真实存在的后缀来选择扩展名，按 png → jpg → webp 顺序优先
        candidate_exts = [".png", ".jpg", ".webp"]
        chosen_ext = ext
        head_on_disk = os.path.join(base_dir, head)
        for ce in candidate_exts:
            if os.path.exists(os.path.join(head_on_disk, f"{frame_range}{ce}")):
                chosen_ext = ce
                break
        if not chosen_ext:
            chosen_ext = ".png"

        new_tail = f"{frame_range}{chosen_ext}"
        new_fp = os.path.join(head, new_tail).replace("\\", "/")
        fr["file_path"] = new_fp
    else:
        fr["file_path"] = fp

with open(transforms_path, "w") as f:
    json.dump(data, f, indent=4)
PY
  fi

  # 逐帧拷贝的目录：
  # - fmasks / images / images_alpha / poses_2d / poses_3d / poses_sapiens / skeletons
  # 结构为：dir_name/xx/@frame_range@.webp 或 .json 等
  local per_frame_dirs=(fmasks images images_alpha poses_2d poses_3d poses_sapiens skeletons)

  shopt -s nullglob
  for d in "${per_frame_dirs[@]}"; do
    local src_dir="${SOURCE_ROOT}/${d}"
    local dst_dir="${TARGET_ROOT}/${d}"

    if [[ ! -d "${src_dir}" ]]; then
      echo ">> 跳过 ${d}，源目录不存在: ${src_dir}"
      continue
    fi

    echo ">> 复制逐帧目录: ${d}"
    mkdir -p "${dst_dir}"

    # 遍历一级子目录 xx/
    for cam_dir in "${src_dir}"/*; do
      [[ -d "${cam_dir}" ]] || continue
      cam_name="$(basename "${cam_dir}")"
      mkdir -p "${dst_dir}/${cam_name}"

      # 在每个 xx/ 下，只复制 frame_range.* 的文件
      found_any=false
      for f in "${cam_dir}/${FRAME_RANGE}."*; do
        [[ -f "${f}" ]] || continue
        cp "${f}" "${dst_dir}/${cam_name}/"
        found_any=true
      done

      if [[ "${found_any}" == false ]]; then
        echo "   >> 警告: 在 ${cam_dir} 下未找到帧 ${FRAME_RANGE} 对应文件"
      fi
    done
  done
  shopt -u nullglob

  echo ">> DNARender 数据复制完成"
}

# 确保 conda shell 函数可用
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo ">> Error: conda 未安装，无法使用虚拟环境"
  exit 1
fi

echo ">> [DNARender] 复制数据并执行 Diffuman4D inference"
copy_dnarender_data

conda activate diffuman4d
echo ">> Running Diffuman4D inference (DNARender)..."
echo ">> exp=$INFERENCE_EXP data.scene_label=$INFERENCE_SCENE_LABEL data.data_dir=$INFERENCE_DATA_DIR"

INFERENCE_TARGET_DIR="${INFERENCE_RESULT_ROOT}/output/results/${INFERENCE_EXP}/${INFERENCE_SCENE_LABEL}"
if [[ -n "${INFERENCE_RESULT_ROOT:-}" && -n "${INFERENCE_EXP:-}" && -n "${INFERENCE_SCENE_LABEL:-}" ]]; then
  echo ">> (info) 推理结果将写入: $INFERENCE_TARGET_DIR"
else
  echo ">> Warning: INFERENCE_RESULT_ROOT, INFERENCE_EXP, or INFERENCE_SCENE_LABEL is empty, inference path may be invalid."
fi

if [[ -z "${INFERENCE_RESULT_ROOT:-}" ]]; then
  echo ">> Error: INFERENCE_RESULT_ROOT 未配置，无法切换到 Diffuman4D 目录"
  exit 1
fi

DIFFUMAN4D_DIR="${INFERENCE_RESULT_ROOT}"
if [[ ! -d "${DIFFUMAN4D_DIR}" ]]; then
  echo ">> Error: Diffuman4D directory not found: ${DIFFUMAN4D_DIR}"
  exit 1
fi

ORIGINAL_DIR="$(pwd)"
cd "${DIFFUMAN4D_DIR}"
echo ">> Changed to Diffuman4D directory: ${DIFFUMAN4D_DIR}"
python "${INFERENCE_SCRIPT_PATH:-inference.py}" \
  exp="${INFERENCE_EXP}" \
  data.scene_label="${INFERENCE_SCENE_LABEL}" \
  data.data_dir="${INFERENCE_DATA_DIR}"
cd "${ORIGINAL_DIR}"

echo ">> [DNARender] inference 完成"


