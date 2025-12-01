#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR_FALLBACK="$(cd "${SCRIPT_DIR_FALLBACK}/../.." && pwd)"
DEFAULT_CONFIG_FILE="${ROOT_DIR_FALLBACK}/config/Diffuman/Hq_4_2_DNA_48.py"

CONFIG_FILE="$DEFAULT_CONFIG_FILE"
PASSTHRU_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE=$2
      shift 2
      ;;
    *)
      PASSTHRU_ARGS+=("$1")
      shift
      ;;
  esac
done

if ((${#PASSTHRU_ARGS[@]})); then
  set -- "${PASSTHRU_ARGS[@]}"
else
  set --
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi

eval "$(
python - "$CONFIG_FILE" "$ROOT_DIR_FALLBACK" "$SCRIPT_DIR_FALLBACK" <<'PY'
import sys
import os
import importlib.util
config_path, fallback_root, fallback_script = sys.argv[1:4]

def emit(name, value):
    if value is None:
        return
    if isinstance(value, bool):
        value = "true" if value else "false"
    value = str(value).replace("'", "'\"'\"'")
    print(f"{name}='{value}'")

# Load config from Python or YAML file
data = {}
config_module = None
if config_path.endswith('.py'):
    try:
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        data = getattr(config_module, 'config', {})
    except Exception as e:
        print(f"Error loading Python config: {e}", file=sys.stderr)
        data = {}
else:
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    except Exception as e:
        print(f"Error loading YAML config: {e}", file=sys.stderr)
        data = {}

section = data.get("run_sh_config") or {}
# Read inference_result_root from module level if it exists
if config_module is not None:
    emit("RUN_INFERENCE_RESULT_ROOT", getattr(config_module, 'inference_result_root', None))
emit("RUN_ROOT_DIR", section.get("root_dir", fallback_root))
emit("RUN_SCRIPT_DIR", section.get("script_dir", fallback_script))
emit("RUN_DEFAULT_DATADIR", section.get("default_datadir"))
emit("RUN_DEFAULT_PREDICT_KEYPOINTS_VIEW", section.get("default_predict_keypoints_view"))
emit("RUN_DEFAULT_USE_ALIGN_POSE", section.get("default_use_align_pose"))
remove_bg = section.get("remove_background") or {}
emit("RUN_REMOVE_BACKGROUND_MODEL", remove_bg.get("model_name"))
emit("RUN_REMOVE_BACKGROUND_BATCH", remove_bg.get("batch_size"))
align_pose = section.get("align_pose_pcd") or {}
emit("RUN_ALIGN_POSE_TARGET_DIR", align_pose.get("target_dir"))
emit("RUN_ALIGN_POSE_SOURCE_FRAME", align_pose.get("source_frame"))
emit("RUN_ALIGN_POSE_TARGET_FRAME", align_pose.get("target_frame"))
emit("RUN_ALIGN_POSE_RUN_ICP", align_pose.get("run_icp"))
collage = section.get("collage") or {}
emit("RUN_COLLAGE_COMPARISON_DIR", collage.get("comparison_images_dir"))
emit("RUN_COLLAGE_IMAGES_PER_ROW", collage.get("images_per_row"))
inference = section.get("inference") or {}
emit("RUN_INFERENCE_EXP", inference.get("exp"))
emit("RUN_INFERENCE_SCENE_LABEL", inference.get("scene_label"))
emit("RUN_INFERENCE_DATA_DIR", inference.get("data_dir"))
emit("RUN_INFERENCE_SCRIPT_PATH", inference.get("inference_script_path"))
visualize_cameras = section.get("visualize_cameras") or {}
emit("RUN_VISUALIZE_CAMERAS_TRANSFORMS_NAME", visualize_cameras.get("transforms_name"))
emit("RUN_VISUALIZE_CAMERAS_SCALE", visualize_cameras.get("scale"))
ns_train = section.get("ns_train") or {}
emit("RUN_NS_TRAIN_DATA_DIR", ns_train.get("data_dir"))
emit("RUN_NS_TRAIN_OUTPUT_DIR", ns_train.get("output_dir"))
colmap2nerf = data.get("colmap2nerf_args") or {}
emit("RUN_COLMAP2NERF_TARGET_DIR", colmap2nerf.get("target"))
actions_envs = section.get("actions_envs") or {}
for key, value in actions_envs.items():
    if value is None:
        continue
    safe_key = ''.join(c if c.isalnum() or c == '_' else '_' for c in key)
    emit(f"RUN_ACTION_ENV_{safe_key}", value)
PY
)"

ROOT_DIR="${RUN_ROOT_DIR:-$ROOT_DIR_FALLBACK}"
SCRIPT_DIR="${RUN_SCRIPT_DIR:-$SCRIPT_DIR_FALLBACK}"
REPO_ROOT="${ROOT_DIR}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

cd "$ROOT_DIR"

read_index_txt() {
    local datadir="$1"
    local colmap_dir="$(dirname "$datadir")/colmap"
    local index_file="$colmap_dir/index.txt"

    if [ -f "$index_file" ]; then
        local views=$(
            awk -F'-' '
                {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
                    if (length($1) > 0) {
                        printf("%s%s", (NR==1 ? "" : ","), $1)
                    }
                }
            ' "$index_file"
        )
        echo "$views"
    else
        echo ""
    fi
}

to_lower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

bool_to_python() {
  local value
  value=$(to_lower "${1:-false}")
  if [[ "$value" == "true" ]]; then
    printf 'True'
  else
    printf 'False'
  fi
}

safe_remove_dir() {
  local target_dir="$1"
  if [[ -z "$target_dir" ]]; then
    echo ">> Warning: empty path provided to safe_remove_dir, skipping."
    return
  fi
  if [[ "$target_dir" == "/" ]]; then
    echo ">> Warning: refusing to remove root directory."
    return
  fi
  if [[ -d "$target_dir" ]]; then
    echo ">> Removing directory: $target_dir"
    rm -rf "$target_dir"
  else
    echo ">> Directory not found (skip removal): $target_dir"
  fi
}

safe_clear_dir() {
  local target_dir="$1"
  if [[ -z "$target_dir" ]]; then
    echo ">> Warning: empty path provided to safe_clear_dir, skipping."
    return
  fi
  if [[ "$target_dir" == "/" ]]; then
    echo ">> Warning: refusing to clear root directory."
    return
  fi
  if [[ -d "$target_dir" ]]; then
    echo ">> Target directory to clear: $target_dir"
    echo ">> Do you want to clear this directory? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
      echo ">> Clearing directory contents: $target_dir"
      rm -rf "${target_dir:?}"/*
      rm -rf "${target_dir:?}"/.[!.]* "${target_dir:?}"/..?* 2>/dev/null || true
      echo ">> Directory cleared successfully."
    else
      echo ">> Skipping directory clearing (user cancelled)."
    fi
  else
    echo ">> Directory not found (skip clearing): $target_dir"
  fi
}

update_source_labels() {
  IFS=',' read -r -a SOURCE_LABELS <<< "$predict_keypoints_view"
}

update_path_vars() {
  KP2D_DIR="$DATADIR/poses_sapiens$HQ_SUFFIX"
  KP3D_DIR="$DATADIR/poses_3d$HQ_SUFFIX"
  PCD_DIR="$DATADIR/poses_pcd$HQ_SUFFIX"
  KP2D_PROJ_DIR="$DATADIR/poses_2d$HQ_SUFFIX"
  TRANSFORM_NAME="transforms${HQ_SUFFIX}.json"
  TRANSFORM_PATH="$DATADIR/$TRANSFORM_NAME"
}

count_skeleton_subdirs() {
  local skeleton_dir="$DATADIR/skeletons"
  if [ -d "$skeleton_dir" ]; then
    find "$skeleton_dir" -mindepth 1 -maxdepth 1 -type d | wc -l
  else
    echo 0
  fi
}

update_target_labels() {
  local base_count
  local required_count
  base_count=$(($(count_skeleton_subdirs) - 1))
  required_count=${#SOURCE_LABELS[@]}
  TARGET_LABELS=()
  if [ "$required_count" -eq 0 ]; then
    echo ">> Warning: SOURCE_LABELS empty, defaulting to 4 target labels."
    required_count=4
  fi
  for ((offset = 1; offset <= required_count; offset++)); do
    TARGET_LABELS+=($((base_count + offset)))
  done
}

get_action_env() {
  local action="$1"
  local sanitized="${action//[^a-zA-Z0-9_]/_}"
  local var_name="RUN_ACTION_ENV_${sanitized}"
  local value
  value=$(eval "printf '%s' \"\${$var_name-}\"")
  if [[ -z "$value" ]]; then
    case "$action" in
      colmap_construction) value="colmap-cuda118" ;;
      colmap2nerf) value="nerfstudio" ;;
      predict_keypoints) value="sapiens_lite" ;;
      *) value="diffuman4d" ;;
    esac
  fi
  printf '%s' "$value"
}

print_action_io() {
  local action="$1"
  case "$action" in
    remove_background)
      echo ">> IO [$action]: inputs=images:$DATADIR/images; outputs=fmasks:$DATADIR/fmasks"
      ;;
    predict_keypoints)
      echo ">> IO [$action]: inputs=images:$DATADIR/images,fmasks:$DATADIR/fmasks; outputs=kp2d:$KP2D_DIR"
      ;;
    triangulate_skeleton)
      echo ">> IO [$action]: inputs=camera:$TRANSFORM_PATH,kp2d:$KP2D_DIR; outputs=kp3d:$KP3D_DIR,pcd:$PCD_DIR,kp2d_proj:$KP2D_PROJ_DIR"
      ;;
    draw_skeleton)
      echo ">> IO [$action]: input=kp2d_proj:$KP2D_PROJ_DIR; output=skeletons_gt:$DATADIR/skeletons_gt"
      ;;
    align_pose_pcd)
      echo ">> IO [$action]: source=$DATADIR target=$ALIGN_POSE_TARGET_DIR output=$DATADIR"
      ;;
    project_pcd_skeletons)
      echo ">> IO [$action]: inputs=pcd:$DATADIR/poses_pcd,camera:$DATADIR/transforms.json; output=skeletons:$DATADIR/skeletons"
      ;;
    overlay_skeletons)
      echo ">> IO [$action]: inputs=images:$DATADIR/images,skeletons:$DATADIR/skeletons; outputs=overlay_project:$DATADIR/overlay_project_pcd,overlay_poses:$DATADIR/overlay_poses_sapiens"
      ;;
    collage)
      echo ">> IO [$action]: inputs=skeletons:$DATADIR/skeletons,comparison:$COLLAGE_COMPARISON_DIR; output=${DATADIR}/skeletons_collage.webp"
      ;;
    sync_cameras)
      echo ">> IO [$action]: operating_on=$DATADIR subdirs=images,fmasks,skeletons_gt:skeletons"
      ;;
    inference)
      local inference_target_dir="${INFERENCE_RESULT_ROOT}/${INFERENCE_EXP}/${INFERENCE_SCENE_LABEL}"
      echo ">> IO [$action]: input=$INFERENCE_DATA_DIR; output=$inference_target_dir"
      ;;
    ns_train)
      echo ">> IO [$action]: data=$NS_TRAIN_DATA_DIR"
      ;;
    colmap2nerf)
      echo ">> IO [$action]: target_dir=$COLMAP2NERF_TARGET_DIR"
      ;;
    colmap_construction)
      echo ">> IO [$action]: data_root=$DATADIR"
      ;;
    *)
      echo ">> IO [$action]: (no explicit IO paths configured)"
      ;;
  esac
}

predict_keypoints_view="${RUN_DEFAULT_PREDICT_KEYPOINTS_VIEW:?run_sh_config.default_predict_keypoints_view 未配置}"
USE_ALIGN_POSE="${RUN_DEFAULT_USE_ALIGN_POSE:?run_sh_config.default_use_align_pose 未配置}"
REMOVE_BACKGROUND_MODEL_NAME="${RUN_REMOVE_BACKGROUND_MODEL:?run_sh_config.remove_background.model_name 未配置}"
REMOVE_BACKGROUND_BATCH="${RUN_REMOVE_BACKGROUND_BATCH:?run_sh_config.remove_background.batch_size 未配置}"
ALIGN_POSE_TARGET_DIR="${RUN_ALIGN_POSE_TARGET_DIR:?run_sh_config.align_pose_pcd.target_dir 未配置}"
ALIGN_POSE_SOURCE_FRAME="${RUN_ALIGN_POSE_SOURCE_FRAME:?run_sh_config.align_pose_pcd.source_frame 未配置}"
ALIGN_POSE_TARGET_FRAME="${RUN_ALIGN_POSE_TARGET_FRAME:?run_sh_config.align_pose_pcd.target_frame 未配置}"
ALIGN_POSE_RUN_ICP="${RUN_ALIGN_POSE_RUN_ICP:?run_sh_config.align_pose_pcd.run_icp 未配置}"
COLLAGE_COMPARISON_DIR="${RUN_COLLAGE_COMPARISON_DIR:?run_sh_config.collage.comparison_images_dir 未配置}"
COLLAGE_IMAGES_PER_ROW="${RUN_COLLAGE_IMAGES_PER_ROW:?run_sh_config.collage.images_per_row 未配置}"
INFERENCE_EXP="${RUN_INFERENCE_EXP:?run_sh_config.inference.exp 未配置}"
INFERENCE_SCENE_LABEL="${RUN_INFERENCE_SCENE_LABEL:?run_sh_config.inference.scene_label 未配置}"
INFERENCE_DATA_DIR="${RUN_INFERENCE_DATA_DIR:?run_sh_config.inference.data_dir 未配置}"
INFERENCE_RESULT_ROOT="${RUN_INFERENCE_RESULT_ROOT:?inference_result_root 未配置}"
INFERENCE_SCRIPT_PATH="${RUN_INFERENCE_SCRIPT_PATH:-inference.py}"
VISUALIZE_CAMERAS_TRANSFORMS_NAME="${RUN_VISUALIZE_CAMERAS_TRANSFORMS_NAME:-transforms}"
VISUALIZE_CAMERAS_SCALE="${RUN_VISUALIZE_CAMERAS_SCALE:-0.2}"
DEFAULT_DATADIR="${RUN_DEFAULT_DATADIR:?run_sh_config.default_datadir 未配置}"
NS_TRAIN_DATA_DIR="${RUN_NS_TRAIN_DATA_DIR:?run_sh_config.ns_train.data_dir 未配置}"
NS_TRAIN_OUTPUT_DIR="${RUN_NS_TRAIN_OUTPUT_DIR:-}"
COLMAP2NERF_TARGET_DIR="${RUN_COLMAP2NERF_TARGET_DIR:?colmap2nerf_args.target 未配置}"

USE_ALIGN_POSE=$(to_lower "$USE_ALIGN_POSE")
DATADIR="$DEFAULT_DATADIR"
HQ_SUFFIX=""
SOURCE_LABELS=()
TARGET_LABELS=()
ACTIONS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATADIR=$2
      shift 2
      ;;
    --actions)
      IFS=',' read -r -a ACTIONS <<< "$2"
      shift 2
      ;;
    --ns_train_output_dir)
      NS_TRAIN_OUTPUT_DIR=$2
      shift 2
      ;;
    --use_align_pose)
      USE_ALIGN_POSE=$(to_lower "$2")
      shift 2
      ;;
    --*)
      echo ">> Unknown arg: $1"; exit 1;;
    *)
      ACTIONS+=("$1")
      shift
      ;;
  esac
done

if [[ "$USE_ALIGN_POSE" != "true" && "$USE_ALIGN_POSE" != "false" ]]; then
  echo ">> Error: --use_align_pose expects true or false."
  exit 1
fi

if [ "$USE_ALIGN_POSE" = "true" ]; then
  HQ_SUFFIX="_HQ"
else
  HQ_SUFFIX=""
fi

update_path_vars
update_source_labels

if [ -n "$DATADIR" ]; then
  index_views=$(read_index_txt "$DATADIR")
  if [ -n "$index_views" ]; then
    predict_keypoints_view="$index_views"
    echo ">> Auto-loaded predict_keypoints_view: $predict_keypoints_view"
    update_source_labels
  fi
fi

if [ "$USE_ALIGN_POSE" = "true" ]; then
  PREP_ACTIONS=("remove_background" "predict_keypoints" "triangulate_skeleton" "draw_skeleton" "align_pose_pcd" "project_pcd_skeletons" "collage" "sync_cameras")
  # PREP_ACTIONS=("collage" "sync_cameras")
else
  PREP_ACTIONS=("remove_background" "predict_keypoints" "triangulate_skeleton" "draw_skeleton" "project_pcd_skeletons")
fi

# ALL_ACTIONS=("colmap_construction" "colmap2nerf" "${PREP_ACTIONS[@]}" "inference" "ns_train")
ALL_ACTIONS=("ns_train")
# 

if [ ${#ACTIONS[@]} -eq 0 ]; then
  ACTIONS=("${ALL_ACTIONS[@]}")
fi

if [ "$USE_ALIGN_POSE" != "true" ]; then
  FILTERED_ACTIONS=()
  for action in "${ACTIONS[@]}"; do
    if [ "$action" = "align_pose_pcd" ]; then
      echo ">> Skipping align_pose_pcd because --use_align_pose=false"
      continue
    fi
    FILTERED_ACTIONS+=("$action")
  done
  ACTIONS=("${FILTERED_ACTIONS[@]}")
fi

echo ">> Config file: $CONFIG_FILE"
echo ">> Data directory: $DATADIR"
echo ">> Actions: ${ACTIONS[*]}"
echo ">> Use align_pose_pcd: $USE_ALIGN_POSE (suffix: ${HQ_SUFFIX:-none})"

for act in "${ACTIONS[@]}"; do
  print_action_io "$act"
  echo ">> Running skeleton generation action: $act"
  case "$act" in
    colmap_construction)
      echo ">> Running colmap_construction..."
      conda activate "$(get_action_env 'colmap_construction')"
      python "${SCRIPT_DIR}/colmap_construction.py"
      ;;
    colmap2nerf)
      echo ">> Running colmap2nerf..."
      conda activate "$(get_action_env 'colmap2nerf')"
      if [[ -n "$COLMAP2NERF_TARGET_DIR" ]]; then
        safe_remove_dir "$COLMAP2NERF_TARGET_DIR"
      else
        echo ">> Warning: COLMAP2NERF_TARGET_DIR is empty, skipping cleanup."
      fi
      python "${SCRIPT_DIR}/colmap2nerf.py"
      ;;
    remove_background)
      conda activate "$(get_action_env 'remove_background')"
      python "${SCRIPT_DIR}/preprocess/remove_background.py" \
        --images_dir "$DATADIR/images" \
        --out_fmasks_dir "$DATADIR/fmasks" \
        --model_name "$REMOVE_BACKGROUND_MODEL_NAME" \
        --batch_size "$REMOVE_BACKGROUND_BATCH"
      ;;
    predict_keypoints)
      conda activate "$(get_action_env 'predict_keypoints')"
      python "${SCRIPT_DIR}/preprocess/predict_keypoints.py" \
        --images_dir "$DATADIR/images" \
        --fmasks_dir "$DATADIR/fmasks" \
        --out_kp2d_dir "$KP2D_DIR"
      ;;
    triangulate_skeleton)
      conda activate "$(get_action_env 'triangulate_skeleton')"
      python "${SCRIPT_DIR}/preprocess/triangulate_skeleton.py" \
        --camera_path "$TRANSFORM_PATH" \
        --kp2d_dir "$KP2D_DIR" \
        --out_kp3d_dir "$KP3D_DIR" \
        --out_pcd_dir "$PCD_DIR" \
        --out_kp2d_proj_dir "$KP2D_PROJ_DIR" \
        --process_subdirs "$predict_keypoints_view"
      ;;
    draw_skeleton)
      conda activate "$(get_action_env 'draw_skeleton')"
      python "${SCRIPT_DIR}/preprocess/draw_skeleton.py" \
        --kp2d_dir "$KP2D_PROJ_DIR" \
        --out_kpmap_dir "$DATADIR/skeletons_gt" \
        --process_subdirs "$predict_keypoints_view"
      ;;
    align_pose_pcd)
      if [ "$USE_ALIGN_POSE" != "true" ]; then
        echo ">> Skipping align_pose_pcd (disabled)."
        continue
      fi
      conda activate "$(get_action_env 'align_pose_pcd')"
      python "${SCRIPT_DIR}/preprocess/align_pose_pcd.py" \
        --source_dir "$DATADIR" \
        --target_dir "$ALIGN_POSE_TARGET_DIR" \
        --source_frame "$ALIGN_POSE_SOURCE_FRAME" \
        --target_frame "$ALIGN_POSE_TARGET_FRAME" \
        --transform_name "$TRANSFORM_NAME" \
        --output_dir "$DATADIR" \
        --run_icp "$(bool_to_python "$ALIGN_POSE_RUN_ICP")"
      ;;
    project_pcd_skeletons)
      conda activate "$(get_action_env 'project_pcd_skeletons')"
      python "${SCRIPT_DIR}/preprocess/project_pcd_skeletons.py" \
        --pcd_dir "$DATADIR/poses_pcd" \
        --camera_path "$DATADIR/transforms.json" \
        --out_kpmap_dir "$DATADIR/skeletons" \
        --skip_exists False
      ;;
    overlay_skeletons)
      conda activate "$(get_action_env 'overlay_skeletons')"
      python "${SCRIPT_DIR}/preprocess/overlay_skeletons.py" \
        --datadir "$DATADIR" \
        --images_dir "$DATADIR/images" \
        --project_skeleton_dir "$DATADIR/skeletons" \
        --poses_skeleton_dir "$KP2D_DIR" \
        --project_overlay_dir "$DATADIR/overlay_project_pcd" \
        --poses_overlay_dir "$DATADIR/overlay_poses_sapiens" \
        --collage_output_path "$DATADIR/skeleton_overlay_collage.png"
      ;;
    collage)
      if [ "$USE_ALIGN_POSE" != "true" ]; then
        echo ">> Skipping collage (align_pose_pcd disabled)."
        continue
      fi
      if [ -d "$DATADIR/skeletons" ]; then
        echo ">> Creating skeleton collage..."
        conda activate "$(get_action_env 'collage')"
        update_target_labels
        exclude_args=()
        if [ ${#TARGET_LABELS[@]} -gt 0 ]; then
          exclude_args=(--exclude_labels "${TARGET_LABELS[@]}")
        fi
        python "${SCRIPT_DIR}/preprocess/collage_skeletons.py" \
          --images_dir "$DATADIR/skeletons" \
          --comparison_images_dir "$COLLAGE_COMPARISON_DIR" \
          --output_path "$DATADIR/skeletons_collage.webp" \
          --images_per_row "$COLLAGE_IMAGES_PER_ROW" \
          "${exclude_args[@]}"
      else
        echo ">> Skip collage: skeletons directory not found."
      fi
      ;;
    sync_cameras)
      if [ "$USE_ALIGN_POSE" != "true" ]; then
        echo ">> Skipping sync_cameras (align_pose_pcd disabled)."
        continue
      fi
      echo ">> Synchronizing camera folders..."
      conda activate "$(get_action_env 'sync_cameras')"
      update_target_labels
      python "${SCRIPT_DIR}/preprocess/sync_cameras.py" \
        --data_dir "$DATADIR" \
        --source_labels "${SOURCE_LABELS[@]}" \
        --target_labels "${TARGET_LABELS[@]}" \
        --subdirs images fmasks skeletons_gt:skeletons
      ;;
    inference)
      conda activate "$(get_action_env 'inference')"
      echo ">> Running inference..."
      echo ">> exp=$INFERENCE_EXP data.scene_label=$INFERENCE_SCENE_LABEL data.data_dir=$INFERENCE_RESULT_ROOT"
      INFERENCE_TARGET_DIR="${INFERENCE_RESULT_ROOT}/${INFERENCE_EXP}/${INFERENCE_SCENE_LABEL}"
      if [[ -n "$INFERENCE_RESULT_ROOT" && -n "$INFERENCE_EXP" && -n "$INFERENCE_SCENE_LABEL" ]]; then
        safe_clear_dir "$INFERENCE_TARGET_DIR"
      else
        echo ">> Warning: INFERENCE_RESULT_ROOT, INFERENCE_EXP, or INFERENCE_SCENE_LABEL is empty, skipping directory clearing."
      fi
      # Switch to Diffuman4D project directory (taken from inference_result_root in config)
      DIFFUMAN4D_DIR="$INFERENCE_RESULT_ROOT"
      if [[ ! -d "$DIFFUMAN4D_DIR" ]]; then
        echo ">> Error: Diffuman4D directory not found: $DIFFUMAN4D_DIR"
        exit 1
      fi
      ORIGINAL_DIR="$(pwd)"
      cd "$DIFFUMAN4D_DIR"
      echo ">> Changed to Diffuman4D directory: $DIFFUMAN4D_DIR"
      python "$INFERENCE_SCRIPT_PATH" \
        exp="$INFERENCE_EXP" \
        data.scene_label="$INFERENCE_SCENE_LABEL" \
        data.data_dir="$INFERENCE_DATA_DIR"
      cd "$ORIGINAL_DIR"
      ;;
    ns_train)
      # Copy poses_pcd/000000.ply to inference output directory as sparse_pcd.ply
      # Use NS_TRAIN_DATA_DIR which is already the full inference_result_path from config
      if [[ -n "$NS_TRAIN_DATA_DIR" ]]; then
        SOURCE_PLY="$DATADIR/poses_pcd/000000.ply"
        TARGET_PLY="$NS_TRAIN_DATA_DIR/sparse_pcd.ply"
        echo ">> Attempting to copy PLY file..."
        echo ">> Source: $SOURCE_PLY"
        echo ">> Target: $TARGET_PLY"
        if [[ -f "$SOURCE_PLY" ]]; then
          mkdir -p "$NS_TRAIN_DATA_DIR"
          cp "$SOURCE_PLY" "$TARGET_PLY"
          echo ">> Successfully copied $SOURCE_PLY to $TARGET_PLY"
        else
          echo ">> Warning: Source PLY file not found: $SOURCE_PLY"
          echo ">> DATADIR is: $DATADIR"
        fi
      else
        echo ">> Warning: NS_TRAIN_DATA_DIR is not configured, skipping PLY copy"
      fi
      
      # Visualize cameras before training
      if [[ -n "$NS_TRAIN_DATA_DIR" ]]; then
        TRANSFORMS_JSON="${NS_TRAIN_DATA_DIR}/${VISUALIZE_CAMERAS_TRANSFORMS_NAME}.json"
        OUTPUT_PLY="${NS_TRAIN_DATA_DIR}/${VISUALIZE_CAMERAS_TRANSFORMS_NAME}.ply"
        
        if [[ -f "$TRANSFORMS_JSON" ]]; then
          echo ">> Visualizing cameras from $TRANSFORMS_JSON"
          conda activate "$(get_action_env 'colmap2nerf')"
          python "${SCRIPT_DIR}/visualize_cameras_as_ply.py" \
            --transforms_json "$TRANSFORMS_JSON" \
            --output_file "$OUTPUT_PLY" \
            --scale "$VISUALIZE_CAMERAS_SCALE"
          echo ">> Camera visualization saved to $OUTPUT_PLY"
        else
          echo ">> Warning: Transforms JSON not found: $TRANSFORMS_JSON, skipping camera visualization"
        fi
      else
        echo ">> Warning: NS_TRAIN_DATA_DIR is not configured, skipping camera visualization"
      fi
      
      if [ -d "$NS_TRAIN_DATA_DIR" ]; then
        echo ">> Running ns-train splatfacto..."
        conda activate "$(get_action_env 'colmap2nerf')"
        ns_train_cmd=(ns-train splatfacto --data "$NS_TRAIN_DATA_DIR")
        if [[ -n "$NS_TRAIN_OUTPUT_DIR" ]]; then
          echo ">> ns-train output_dir: $NS_TRAIN_OUTPUT_DIR"
          ns_train_cmd+=(--output-dir "$NS_TRAIN_OUTPUT_DIR")
        fi
        "${ns_train_cmd[@]}"
      else
        echo ">> Skip ns-train: data directory not found: $NS_TRAIN_DATA_DIR"
      fi
      ;;
    *)
      echo ">> Invalid action: $act"
      echo ">> Valid actions: ${ALL_ACTIONS[*]}"
      exit 1
      ;;
  esac
done