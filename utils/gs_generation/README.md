# run.sh 使用说明

## 概述

`run.sh` 是一个用于生成高斯点云（Gaussian Splatting）的完整流水线脚本。它自动化了从图像预处理到模型推理和训练的整个流程。

## 使用方法

### 基本用法

```bash
./run.sh [选项] [动作列表]
```

### 命令行参数

- `--config <配置文件路径>`: 指定配置文件（Python），默认为 `config/Diffuman/Hq_4_2_DNA_48.py`
- `--data_dir <数据目录>`: 指定数据目录路径
- `--actions <动作列表>`: 指定要执行的动作，多个动作用逗号分隔，例如：`--actions colmap_construction,inference`
- `--use_align_pose <true|false>`: 是否使用姿态对齐功能（影响输出文件的后缀，true 时使用 `_HQ` 后缀）

### 示例

```bash
# 运行所有步骤
bash utils/gs_generation/run.sh

## 配置说明

脚本从配置文件中读取参数，包含以下配置项：

- `default_datadir`: 默认数据目录
- `default_predict_keypoints_view`: 默认的关键点预测视图
- `default_use_align_pose`: 是否使用姿态对齐
- `remove_background`: 背景移除配置（model_name, batch_size）
- `align_pose_pcd`: 姿态对齐配置（target_dir, source_frame, target_frame, run_icp）
- `collage`: 拼贴配置（comparison_images_dir, images_per_row）
- `inference`: 推理配置（exp, scene_label, data_dir, inference_script_path）
- `ns_train`: 训练配置（data_dir）
- `actions_envs`: 各动作使用的 conda 环境配置

## 完整流程步骤详解

### 1. 配置加载阶段

**功能**: 加载配置文件并初始化环境变量

**执行内容**:
- 解析命令行参数
- 加载 Python 或 YAML 配置文件
- 初始化 conda 环境
- 设置 PYTHONPATH
- 读取配置中的各项参数

**输出**: 设置所有必要的环境变量和路径变量

---

### 2. colmap_construction（COLMAP 构建）

**功能**: 使用 COLMAP 进行相机姿态估计和稀疏重建

**执行内容**:
- 激活 conda 环境（默认：`colmap-cuda118`）
- 运行 `colmap_construction.py`
- 生成相机参数和稀疏点云

**输入**: 图像数据
**输出**: COLMAP 重建结果（相机参数、稀疏点云等）

**依赖**: 需要先准备好图像数据

---

### 3. colmap2nerf（COLMAP 转 NeRF 格式）

**功能**: 将 COLMAP 结果转换为 NeRF Studio 格式

**执行内容**:
- 激活 conda 环境（默认：`nerfstudio`）
- 清空目标目录（如果配置了 `COLMAP2NERF_TARGET_DIR`）
- 运行 `colmap2nerf.py`
- 生成 NeRF Studio 格式的 transforms.json

**输入**: COLMAP 重建结果
**输出**: NeRF Studio 格式的相机参数文件

**依赖**: 需要先完成 `colmap_construction`

---

### 4. remove_background（背景移除）

**功能**: 从图像中移除背景，生成前景掩码

**执行内容**:
- 激活 conda 环境（从配置读取，默认：`diffuman4d`）
- 运行 `preprocess/remove_background.py`
- 处理 `$DATADIR/images` 目录下的所有图像
- 生成前景掩码到 `$DATADIR/fmasks` 目录

**输入**: 
- `$DATADIR/images`: 原始图像
- 配置中的模型名称和批次大小

**输出**: 
- `$DATADIR/fmasks`: 前景掩码图像

**依赖**: 需要先准备好图像数据

---

### 5. predict_keypoints（关键点预测）

**功能**: 使用 Sapiens 模型预测人体关键点（2D）

**执行内容**:
- 激活 conda 环境（默认：`sapiens_lite`）
- 运行 `preprocess/predict_keypoints.py`
- 处理图像和前景掩码
- 生成 2D 关键点到 `$DATADIR/poses_sapiens` 或 `$DATADIR/poses_sapiens_HQ`（取决于 `use_align_pose`）

**输入**: 
- `$DATADIR/images`: 图像
- `$DATADIR/fmasks`: 前景掩码

**输出**: 
- `$DATADIR/poses_sapiens[_HQ]`: 2D 关键点数据

**依赖**: 需要先完成 `remove_background`

---

### 6. triangulate_skeleton（骨架三角化）

**功能**: 将 2D 关键点三角化为 3D 骨架

**执行内容**:
- 激活 conda 环境（默认：`diffuman4d`）
- 运行 `preprocess/triangulate_skeleton.py`
- 使用相机参数将 2D 关键点三角化为 3D
- 生成 3D 关键点、点云和投影后的 2D 关键点

**输入**: 
- `$TRANSFORM_PATH`: 相机参数文件（transforms.json 或 transforms_HQ.json）
- `$KP2D_DIR`: 2D 关键点数据
- `predict_keypoints_view`: 要处理的视图列表

**输出**: 
- `$DATADIR/poses_3d[_HQ]`: 3D 关键点
- `$DATADIR/poses_pcd[_HQ]`: 点云数据
- `$DATADIR/poses_2d[_HQ]`: 投影后的 2D 关键点

**依赖**: 需要先完成 `predict_keypoints` 和 `colmap2nerf`

---

### 7. draw_skeleton（绘制骨架）

**功能**: 在图像上绘制骨架关键点

**执行内容**:
- 激活 conda 环境（默认：`diffuman4d`）
- 运行 `preprocess/draw_skeleton.py`
- 在投影后的 2D 关键点上绘制骨架
- 生成骨架可视化图像

**输入**: 
- `$KP2D_PROJ_DIR`: 投影后的 2D 关键点

**输出**: 
- `$DATADIR/skeletons_gt`: 骨架可视化图像

**依赖**: 需要先完成 `triangulate_skeleton`

---

### 8. align_pose_pcd（姿态对齐点云）

**功能**: HQ相机阵列（4视角）对齐到DNA-Render相机阵列(48视角)

**执行内容**:
- 仅在 `--use_align_pose=true` 时执行
- 激活 conda 环境（默认：`diffuman4d`）
- 运行 `preprocess/align_pose_pcd.py`
- 将源帧的点云对齐到目标帧

**输入**: 
- `$DATADIR`: 源数据目录
- `$ALIGN_POSE_TARGET_DIR`: 目标数据目录
- `$ALIGN_POSE_SOURCE_FRAME`: 源帧
- `$ALIGN_POSE_TARGET_FRAME`: 目标帧
- `$TRANSFORM_NAME`: 变换文件名

**输出**: 
- 对齐后的点云数据（带 `_HQ` 后缀）

**依赖**: 需要先完成 `triangulate_skeleton`

**注意**: 此步骤会生成带 `_HQ` 后缀的文件，影响后续步骤的输出路径

---

### 9. project_pcd_skeletons（投影点云骨架）

**功能**: 将点云骨架投影到图像平面

**执行内容**:
- 激活 conda 环境（默认：`diffuman4d`）
- 运行 `preprocess/project_pcd_skeletons.py`
- 将 3D 点云骨架投影到各个相机视角（DNA-Render）
- 生成投影后的骨架图像

**输入**: 
- `$DATADIR/poses_pcd`: 点云数据
- `$DATADIR/transforms.json`: 相机参数

**输出**: 
- `$DATADIR/skeletons`: 投影后的骨架图像

**依赖**: 需要先完成 `triangulate_skeleton`

---

### 10. collage（拼贴）

**功能**: 创建骨架图像的拼贴图

**执行内容**:
- 仅在 `--use_align_pose=true` 时执行
- 激活 conda 环境（从配置读取，默认：`diffuman4d`）
- 运行 `preprocess/collage_skeletons.py`
- 将骨架图像和对比图像拼贴在一起
- 生成拼贴图像

**输入**: 
- `$DATADIR/skeletons`: 骨架图像目录
- `$COLLAGE_COMPARISON_DIR`: 对比图像目录
- `$COLLAGE_IMAGES_PER_ROW`: 每行图像数量

**输出**: 
- `$DATADIR/skeletons_collage.webp`: 拼贴图像

**依赖**: 需要先完成 `project_pcd_skeletons`

---

### 11. sync_cameras（同步相机）

**功能**: 同步不同标签的相机文件夹

**执行内容**:
- 仅在 `--use_align_pose=true` 时执行
- 激活 conda 环境（默认：`diffuman4d`）
- 运行 `preprocess/sync_cameras.py`
- 将源标签的相机数据同步到目标标签
- 同步 images、fmasks、skeletons_gt 等子目录

**输入**: 
- `$DATADIR`: 数据目录
- `SOURCE_LABELS`: 源标签列表
- `TARGET_LABELS`: 目标标签列表（自动计算）

**输出**: 
- 同步后的目标标签目录

**依赖**: 需要先完成 `draw_skeleton` 和 `project_pcd_skeletons`

---

### 12. overlay_skeletons（骨架叠加）

**功能**: 将骨架叠加到原始图像上

**执行内容**:
- 激活 conda 环境（默认：`diffuman4d`）
- 运行 `preprocess/overlay_skeletons.py`
- 将投影骨架和姿态骨架叠加到原始图像
- 生成叠加图像和拼贴图

**输入**: 
- `$DATADIR/images`: 原始图像
- `$DATADIR/skeletons`: 投影骨架
- `$KP2D_DIR`: 姿态骨架

**输出**: 
- `$DATADIR/overlay_project_pcd`: 投影骨架叠加图像
- `$DATADIR/overlay_poses_sapiens`: 姿态骨架叠加图像
- `$DATADIR/skeleton_overlay_collage.png`: 叠加拼贴图

**依赖**: 需要先完成 `project_pcd_skeletons` 和 `predict_keypoints`

---
### 13. inference（推理）

**功能**: 运行 Diffuman4D 模型推理

**执行内容**:
- 激活 conda 环境（默认：`diffuman4d`）
- **清空目标目录**: 在执行推理前，会提示用户确认是否清空 `${INFERENCE_DATA_DIR}/${INFERENCE_EXP}/${INFERENCE_SCENE_LABEL}` 目录
  - 显示目标目录路径
  - 询问用户确认（y/n）
  - 如果用户输入 `y`，则清空目录内容
  - 如果用户输入其他，则跳过清空
- 运行 `/mnt/cvda/cvda_phava/code/Han/Diffuman4D/inference.py`
- 使用配置中的实验名称、场景标签和数据目录

**输入**: 
- `$INFERENCE_DATA_DIR`: 推理数据目录
- `$INFERENCE_EXP`: 实验名称
- `$INFERENCE_SCENE_LABEL`: 场景标签

**输出**: 
- 推理结果保存到 `${INFERENCE_DATA_DIR}/${INFERENCE_EXP}/${INFERENCE_SCENE_LABEL}`

**依赖**: 需要先完成预处理步骤

**注意**: 
- 推理前会清空输出目录，需要用户确认
- 确保配置中的路径正确

---

### 14. ns_train（NeRF Studio 训练）

**功能**: 使用 NeRF Studio 训练高斯点云模型

**执行内容**:
- 检查训练数据目录是否存在
- 激活 conda 环境（默认：`nerfstudio`）
- 运行 `ns-train splatfacto` 命令
- 训练高斯点云模型

**输入**: 
- `$NS_TRAIN_DATA_DIR`: 训练数据目录（NeRF Studio 格式）

**输出**: 
- 训练好的高斯点云模型

**依赖**: 需要先完成 `colmap2nerf` 和预处理步骤

---

## 辅助函数说明

### safe_remove_dir
安全删除目录函数，包含安全检查（防止删除根目录等）

### safe_clear_dir
安全清空目录内容函数，包含：
- 安全检查（防止清空根目录）
- 交互式确认（显示路径，询问用户 y/n）
- 清空所有文件和隐藏文件

### update_path_vars
根据 `USE_ALIGN_POSE` 设置更新路径变量（添加或移除 `_HQ` 后缀）

### update_source_labels
从 `predict_keypoints_view` 更新源标签列表

### update_target_labels
根据骨架子目录数量自动计算目标标签

### get_action_env
获取指定动作使用的 conda 环境名称（可从配置中覆盖）

---

## 工作流程示例

```bash
./run.sh --use_align_pose true
```

执行顺序：
1. colmap_construction
2. colmap2nerf
3. remove_background
4. predict_keypoints
5. triangulate_skeleton
6. draw_skeleton
7. align_pose_pcd（生成 `_HQ` 后缀文件）
8. project_pcd_skeletons
9. collage
10. sync_cameras
11. inference
12. ns_train
13. ns-export gaussian-splat --load-config /mnt/cvda/cvda_phava/code/Han/Diffuman4D/outputs/nerfstudio_4_2_48/splatfacto/2025-11-27_195019(非固定，路径下查找)/config.yml --output-dir /mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/nerfstudio_4_2_48（conda activate nerfstudio）
---
