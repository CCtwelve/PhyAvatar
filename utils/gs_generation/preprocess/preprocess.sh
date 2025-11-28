#!/bin/bash
# require miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Function to read index.txt and set predict_keypoints_view
read_index_txt() {
    local datadir="$1"
    local colmap_dir="$(dirname "$datadir")/colmap"
    local index_file="$colmap_dir/index.txt"

    if [ -f "$index_file" ]; then
        # 取 “09 - 127” 左侧的编号，去掉空格，再用逗号拼接
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

# Default value (will be overridden if index.txt exists)
# predict_keypoints_view="04,05,08,09,10"
predict_keypoints_view="09,04,05,17"

# Set USE_ALIGN_POSE to false if not using DNA-Render cameras.
USE_ALIGN_POSE="true"



DATADIR="/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/nerfstudio_4_2_48"

echo ">> Default predict_keypoints_view: $predict_keypoints_view"
# Keep a shell array version for arguments that expect space-separated views
SOURCE_LABELS=()
update_source_labels() {
  IFS=',' read -r -a SOURCE_LABELS <<< "$predict_keypoints_view"
}
update_source_labels

to_lower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
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

TARGET_LABELS=()
update_target_labels() {
  local base_count
  local required_count
  base_count=$(count_skeleton_subdirs)
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
    --use_align_pose)
      USE_ALIGN_POSE=$(to_lower "$2")
      shift 2
      ;;
    *)
      echo ">> Unknown arg: $1"; exit 1;;
  esac
done

# validate use_align_pose flag
if [[ "$USE_ALIGN_POSE" != "true" && "$USE_ALIGN_POSE" != "false" ]]; then
  echo ">> Error: --use_align_pose expects true or false (case-insensitive)."
  exit 1
fi

if [ "$USE_ALIGN_POSE" = "true" ]; then
  HQ_SUFFIX="_HQ"
else
  HQ_SUFFIX=""
fi

update_path_vars

if [ "$USE_ALIGN_POSE" = "true" ]; then
  ALL_ACTIONS=("remove_background" "predict_keypoints" "triangulate_skeleton" "draw_skeleton" "align_pose_pcd" "project_pcd_skeletons")
else
  # ALL_ACTIONS=("remove_background" "predict_keypoints" "triangulate_skeleton" "draw_skeleton" "project_pcd_skeletons")
  ALL_ACTIONS=("remove_background" "predict_keypoints" "triangulate_skeleton" "draw_skeleton" "project_pcd_skeletons")
fi
# ALL_ACTIONS=("align_pose_pcd" "project_pcd_skeletons")

# require a data directory
if [ -z "$DATADIR" ]; then
  echo ">> Error: --data_dir is required"
  exit 1
fi

# Auto-read index.txt and update predict_keypoints_view
if [ -n "$DATADIR" ]; then
  index_views=$(read_index_txt "$DATADIR")
  if [ -n "$index_views" ]; then
    predict_keypoints_view="$index_views"
    echo ">> Auto-loaded predict_keypoints_view from index.txt: $predict_keypoints_view"
    update_source_labels
  fi
fi

# run all actions if not specified
if [ ${#ACTIONS[@]} -eq 0 ]; then
  ACTIONS=("${ALL_ACTIONS[@]}")
elif [ "$USE_ALIGN_POSE" != "true" ]; then
  # Filter out align_pose_pcd if user explicitly listed it
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

echo ">> Data directory: $DATADIR"
echo ">> Actions: ${ACTIONS[@]}"
echo ">> Use align_pose_pcd: $USE_ALIGN_POSE (suffix: ${HQ_SUFFIX:-none})"

for act in "${ACTIONS[@]}"; do
  case "$act" in
    remove_background)
      conda activate diffuman4d
      python scripts/preprocess/remove_background.py \
        --images_dir "$DATADIR/images" \
        --out_fmasks_dir "$DATADIR/fmasks" \
        --model_name ZhengPeng7/BiRefNet \
        --batch_size 8 # decrease it if OOM
      ;;
    predict_keypoints)
      # it is recommend to use a seperate conda environment to run sapiens-lite
      # because sapiens-lite requires pytorch<=2.4.1, https://github.com/open-mmlab/mmdetection/issues/12008
      conda activate sapiens_lite
      python scripts/preprocess/predict_keypoints.py \
        --images_dir "$DATADIR/images" \
        --fmasks_dir "$DATADIR/fmasks" \
        --out_kp2d_dir "$KP2D_DIR"
      ;;
    triangulate_skeleton)
      conda activate diffuman4d
      python scripts/preprocess/triangulate_skeleton.py \
        --camera_path "$TRANSFORM_PATH" \
        --kp2d_dir "$KP2D_DIR" \
        --out_kp3d_dir "$KP3D_DIR" \
        --out_pcd_dir "$PCD_DIR" \
        --out_kp2d_proj_dir "$KP2D_PROJ_DIR" \
        --process_subdirs "$predict_keypoints_view"
      ;;
    draw_skeleton)
      conda activate diffuman4d
      python scripts/preprocess/draw_skeleton.py \
        --kp2d_dir "$KP2D_PROJ_DIR" \
        --out_kpmap_dir "$DATADIR/skeletons_gt"
      ;;
    align_pose_pcd)
      if [ "$USE_ALIGN_POSE" != "true" ]; then
        echo ">> Skipping align_pose_pcd action (disabled)."
        continue
      fi
      conda activate diffuman4d
      python scripts/preprocess_/align_pose_pcd.py \
        --source_dir "$DATADIR" \
        --target_dir "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/data/datasets--krahets--diffuman4d_example/0013_01" \
        --source_frame "000000" \
        --target_frame "000000" \
        --transform_name "$TRANSFORM_NAME" \
        --output_dir "$DATADIR" \
        --run_icp False
      ;;

    project_pcd_skeletons)
      conda activate diffuman4d
      python scripts/preprocess/project_pcd_skeletons.py \
        --pcd_dir "$DATADIR/poses_pcd" \
        --camera_path "$DATADIR/transforms.json" \
        --out_kpmap_dir "$DATADIR/skeletons" \
        --skip_exists False 
      # Optional flags you can uncomment if needed:
      #   --camera_coord_convention opengl
      #   --flip_y True
      ;;
    overlay_skeletons)
      conda activate diffuman4d
      python scripts/preprocess/overlay_skeletons.py \
        --datadir "$DATADIR" \
        --images_dir "$DATADIR/images" \
        --project_skeleton_dir "$DATADIR/skeletons" \
        --poses_skeleton_dir "$KP2D_DIR" \
        --project_overlay_dir "$DATADIR/overlay_project_pcd" \
        --poses_overlay_dir "$DATADIR/overlay_poses_sapiens" \
        --collage_output_path "$DATADIR/skeleton_overlay_collage.png"
      ;;
    *)
      echo "Invalid action: $act" >&2
      exit 1
      ;;
  esac
done

if [ "$USE_ALIGN_POSE" = "true" ]; then
  if [ -d "$DATADIR/skeletons" ]; then
    echo ">> Creating skeleton collage with 8 images per row..."
    conda activate diffuman4d
    python scripts/preprocess_/collage_skeletons.py \
      --images_dir "$DATADIR/skeletons" \
      --comparison_images_dir "/mnt/cvda/cvda_phava/code/Han/Diffuman4D/data/datasets--krahets--diffuman4d_example/0013_01/skeletons" \
      --output_path "$DATADIR/skeletons_collage.webp" \
      --images_per_row 8
  else
    echo ">> Skip collage: skeletons directory not found."
  fi

  echo ">> Synchronizing camera parameters and duplicating camera folders..."
  conda activate diffuman4d
  update_target_labels
  python scripts/preprocess_/sync_cameras.py \
    --data_dir "$DATADIR" \
    --source_labels "${SOURCE_LABELS[@]}" \
    --target_labels "${TARGET_LABELS[@]}" \
    --subdirs images fmasks skeletons_gt:skeletons
else
  echo ">> align_pose_pcd disabled: skipping collage_skeletons.py and sync_cameras.py."
fi
