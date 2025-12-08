#!/usr/bin/env bash
set -euo pipefail

# 先在nerfstudio 环境下执行下面代码
# ns-train splatfacto --data "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/DNARender_0013_01_4_2_48_70"

# DNARender 专用的衣服训练脚本：
# 1. 从 DNARender 数据根目录（diffuman4d_data_root/subject）中复制所需数据到
#    PhyAvatar 工程下的 results/nerfstudio/<filename>（即 soft_path）
# 2. 仅对逐帧子目录（fmasks, images, images_alpha, poses_2d 等）拷贝 frame_range 指定帧
# 3. 复用 run_seg_clothes.sh 中的三个动作：ns_export / data_preparation / ns_train_clothes
#
# 注意：根据用户要求，三个动作（可以理解为“函数”）的逻辑与 run_seg_clothes.sh 保持一致，
# 只通过环境变量/路径来适配 DNARender 数据结构，不修改这三个动作本身的实现。

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
emit("DATA_PREP_SEQUENCE", getattr(mod, "sequence", None))
emit("DATA_PREP_RESOLUTION", getattr(mod, "resolution", None))
emit("FRAME_RANGE", getattr(mod, "frame_range", None))
emit("DIFFUMAN4D_DATA_ROOT", getattr(mod, "diffuman4d_data_root", None))
emit("INFERENCE_SOFT_LINK_PATH", getattr(mod, "inference_soft_link_path", None))
emit("SCENE_FILENAME", getattr(mod, "filename", None))
emit("SOFT_PATH", getattr(mod, "soft_path", None))
emit("INFERENCE_RESULT_PATH", getattr(mod, "inference_result_path", None))
emit("INFERENCE_RESULT_ROOT", getattr(mod, "inference_result_root", None))

cfg = getattr(mod, "config", {})
dp_cfg = cfg.get("data_preparation.py", cfg.get("data_prep_config", {}))

emit("DATA_PREP_IMAGES_DIR", dp_cfg.get("images_dir"))
emit("DATA_PREP_MASKS_DIR", dp_cfg.get("masks_dir"))
emit("DATA_PREP_TRANSFORMS_JSON", dp_cfg.get("transforms_json"))
emit("DATA_PREP_OUTPUT_ROOT", dp_cfg.get("output_root"))
emit("DATA_PREP_SMPLX_ROOT", dp_cfg.get("smplx_root"))
emit("DATA_PREP_ACTORSHQ_SMPLX_ZIP", dp_cfg.get("actorshq_smplx_zip"))
emit("DATA_PREP_ACTORSHQ_SMPLX_DEL_PATH", dp_cfg.get("actorshq_smplx_del_path"))
emit("DATA_PREP_MASKER_PROMPT", dp_cfg.get("masker_prompt"))
emit("DATA_PREP_GENDER", dp_cfg.get("gender"))
emit("DATA_PREP_BOX_THRESHOLD", dp_cfg.get("box_threshold"))
emit("DATA_PREP_TEXT_THRESHOLD", dp_cfg.get("text_threshold"))
emit("DATA_PREP_SKIP_MASKING", dp_cfg.get("skip_masking"))
emit("DATA_PREP_SKIP_REORG", dp_cfg.get("skip_reorg"))
emit("DATA_PREP_COPY_DATA", dp_cfg.get("copy_data"))
emit("DATA_PREP_SKIP_SMPLX", dp_cfg.get("skip_smplx"))
emit("DATA_PREP_SKIP_JSON", dp_cfg.get("skip_json"))
emit("DATA_PREP_SKIP_CLOTH_EXTRACTION", dp_cfg.get("skip_cloth_extraction"))

run_cfg = cfg.get("run_sh_config", {})

# inference 段：供 Diffuman4D 推理使用
inf_cfg = run_cfg.get("inference", {})
emit("INFERENCE_EXP", inf_cfg.get("exp"))
emit("INFERENCE_SCENE_LABEL", inf_cfg.get("scene_label"))
emit("INFERENCE_DATA_DIR", inf_cfg.get("data_dir"))
emit("INFERENCE_SCRIPT_PATH", inf_cfg.get("inference_script_path"))

# ns_train_clothes 段：供衣服 ns-train 使用
ns_train_clothes_cfg = run_cfg.get("ns_train_clothes", {})
emit("NS_TRAIN_CLOTHES_DATA_DIR", ns_train_clothes_cfg.get("data_dir"))
emit("NS_TRAIN_CLOTHES_OUTPUT_DIR", ns_train_clothes_cfg.get("output_dir"))

_ns_train_yaml_path = getattr(mod, "ns_train_yaml_path", None)
_ns_export_gspath = getattr(mod, "ns_export_gspath", None)
emit("NS_TRAIN_YAML_PATH", _ns_train_yaml_path)
emit("NS_EXPORT_GSPATH", _ns_export_gspath)
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

# ---------------------------------------------------------------------------
# 2. 后续流程：与 run_seg_clothes.sh 共用的三个动作
# ---------------------------------------------------------------------------

if [[ -z "${NS_TRAIN_YAML_PATH:-}" || -z "${NS_EXPORT_GSPATH:-}" ]]; then
  echo ">> Error: ns_train_yaml_path 或 ns_export_gspath 未在 ${CONFIG_FILE} 中正确配置"
  exit 1
fi

# 确保 conda shell 函数可用
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo ">> Error: conda 未安装，无法使用虚拟环境"
  exit 1
fi

DATA_PREP_SCRIPT="${ROOT_DIR_FALLBACK}/ActorsHQ-for-Gaussian-Garments/data_preparation.py"

if [[ ! -f "${DATA_PREP_SCRIPT}" ]]; then
  echo ">> Error: data_preparation.py 未找到：${DATA_PREP_SCRIPT}"
  exit 1
fi

# !!! 以下动作（case 分支）逻辑与 run_seg_clothes.sh / run_gen_human.sh 保持一致，只依赖上面的路径环境变量。
# 注意：Diffuman4D 的 inference 和 DNARender 数据复制功能已迁移到 run_gen_human_DNA.sh，
# 这里仅保留 ns_export / data_preparation / ns_train_clothes 三个动作。
ALL_ACTIONS=( "ns_export" "data_preparation" "ns_train_clothes")
for act in "${ALL_ACTIONS[@]}"; do
  echo ">> Running DNARender seg_clothes action: $act"
  case "$act" in
    ns_export)
      echo ">> Activating conda env: nerfstudio"
      conda activate nerfstudio

      echo ">> ns-export gaussian-splat"
      echo "   --load-config '${NS_TRAIN_YAML_PATH}'"
      echo "   --output-dir  '${NS_EXPORT_GSPATH}'"

      ns-export gaussian-splat --load-config "${NS_TRAIN_YAML_PATH}" --output-dir "${NS_EXPORT_GSPATH}"
      ;;

    data_preparation)
      echo ">> Activating conda env: gs2mesh"
      conda activate gs2mesh

      echo ">> data_preparation.py"
      echo "   subject        '${DATA_PREP_SUBJECT:-}'"
      echo "   sequence       '${DATA_PREP_SEQUENCE:-}'"
      echo "   resolution     '${DATA_PREP_RESOLUTION:-}'"
      echo "   images dir     '${DATA_PREP_IMAGES_DIR:-}'"
      echo "   masks dir      '${DATA_PREP_MASKS_DIR:-}'"
      echo "   transforms     '${DATA_PREP_TRANSFORMS_JSON:-}'"
      echo "   output root    '${DATA_PREP_OUTPUT_ROOT:-}'"

      if [[ -z "${DATA_PREP_IMAGES_DIR:-}" || -z "${DATA_PREP_OUTPUT_ROOT:-}" ]]; then
        echo ">> Error: data_prep_config 缺少 images_dir 或 output_root"
        exit 1
      fi

      export DATA_PREP_CONFIG_FILE="${CONFIG_FILE}"
      python "${DATA_PREP_SCRIPT}"
      ;;

    ns_train_clothes)
      echo ">> Activating conda env: nerfstudio"
      conda activate nerfstudio

      if [[ -z "${SOFT_PATH:-}" ]]; then
        echo ">> Error: soft_path 未在 ${CONFIG_FILE} 中正确配置"
        exit 1
      fi

      # data_dir/output_dir 仅通过 ns_train_clothes 配置控制
      DATA_DIR="${NS_TRAIN_CLOTHES_DATA_DIR:-${SOFT_PATH}/seg_clothes}"
      OUTPUT_DIR="${NS_TRAIN_CLOTHES_OUTPUT_DIR:-}"

      if [[ ! -d "${DATA_DIR}" ]]; then
        echo ">> Skip ns-train (clothes): data directory not found: ${DATA_DIR}"
        break
      fi

      echo ">> ns-train splatfacto (clothes)"
      echo "   --data '${DATA_DIR}'"
      if [[ -n "${OUTPUT_DIR}" ]]; then
        echo "   --output-dir '${OUTPUT_DIR}'"
      fi

      ns_train_cmd=(ns-train splatfacto --data "${DATA_DIR}")
      if [[ -n "${OUTPUT_DIR}" ]]; then
        ns_train_cmd+=(--output-dir "${OUTPUT_DIR}")
      fi
      "${ns_train_cmd[@]}"
      ;;

    *)
      echo ">> Unknown action: $act"
      ;;
  esac
done



