#!/usr/bin/env bash
set -euo pipefail

# Very small helper:
# 1. 读取 Hq_4_2_DNA_48.py 里的 ns_train_yaml_path / ns_export_gspath
# 2. 调用：ns-export gaussian-splat --load-config <ns_train_yaml_path> --output-dir <ns_export_gspath>

ROOT_DIR_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${ROOT_DIR_FALLBACK}/config/Diffuman/Hq_4_2_DNA_48.py"

# 允许通过 --config 覆盖配置文件
if [[ "${1-}" == "--config" && $# -ge 2 ]]; then
  CONFIG_FILE=$2
  shift 2
fi

# 从 Python 配置里取出路径
eval "$(
python - "$CONFIG_FILE" <<'PY'
import importlib.util, sys, os

config_path = sys.argv[1]

def emit(name, value):
    if not value:
        return
    value = str(value).replace("'", "'\"'\"'")
    print(f"{name}='{value}'")

spec = importlib.util.spec_from_file_location("cfg", config_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

emit("NS_TRAIN_YAML_PATH", getattr(mod, "ns_train_yaml_path", None))
emit("NS_EXPORT_GSPATH", getattr(mod, "ns_export_gspath", None))

# Base fields
emit("DATA_PREP_SUBJECT", getattr(mod, "subject", None))
emit("DATA_PREP_SEQUENCE", getattr(mod, "sequence", None))
emit("DATA_PREP_RESOLUTION", getattr(mod, "resolution", None))
emit("INFERENCE_RESULT_PATH", getattr(mod, "inference_result_path", None))
emit("INFERENCE_SOFT_LINK_PATH", getattr(mod, "inference_soft_link_path", None))
emit("SCENE_FILENAME", getattr(mod, "filename", None))
emit("SOFT_PATH", getattr(mod, "soft_path", None))

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

# ns-train clothes config（单独的 ns_train_clothes 段）
run_cfg = cfg.get("run_sh_config", {})
ns_train_clothes_cfg = run_cfg.get("ns_train_clothes", {})
emit("NS_TRAIN_CLOTHES_DATA_DIR", ns_train_clothes_cfg.get("data_dir"))
emit("NS_TRAIN_CLOTHES_OUTPUT_DIR", ns_train_clothes_cfg.get("output_dir"))
PY
)"

if [[ -z "${NS_TRAIN_YAML_PATH:-}" || -z "${NS_EXPORT_GSPATH:-}" ]]; then
  echo ">> Error: ns_train_yaml_path 或 ns_export_gspath 未在 ${CONFIG_FILE} 中正确配置"
  exit 1
fi

# Create soft-link from inference_result_path to inference_soft_link_path/filename
if [[ -n "${INFERENCE_RESULT_PATH:-}" && -n "${INFERENCE_SOFT_LINK_PATH:-}" && -n "${SCENE_FILENAME:-}" ]]; then
  LINK_TARGET="${INFERENCE_RESULT_PATH}"
  LINK_DIR="${INFERENCE_SOFT_LINK_PATH}"
  LINK_PATH="${LINK_DIR}/${SCENE_FILENAME}"

  mkdir -p "${LINK_DIR}"
  if [[ -L "${LINK_PATH}" || -e "${LINK_PATH}" ]]; then
    rm -rf "${LINK_PATH}"
  fi
  ln -s "${LINK_TARGET}" "${LINK_PATH}"
  echo ">> Created soft link:"
  echo "   ${LINK_PATH} -> ${LINK_TARGET}"
else
  echo ">> Warning: 无法创建 Diffuman4D 软链接，INFERENCE_RESULT_PATH / INFERENCE_SOFT_LINK_PATH / SCENE_FILENAME 缺失"
fi

# Ensure conda shell functions are available
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

# Actions style (similar to run_gen_human.sh)
ALL_ACTIONS=("ns_export" "data_preparation" "ns_train_clothes")
# ALL_ACTIONS=("ns_train_clothes")
for act in "${ALL_ACTIONS[@]}"; do
  echo ">> Running seg_clothes action: $act"
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


