# 点云配准工具使用说明

## 功能说明

该工具用于将点云C（来自相机阵列A）对齐到点云D（来自相机阵列B）的位置，使得B序列能够正确拍摄到物体。

## 两种配准方法

### 方法1：ICP自动配准（推荐）

**适用场景**：不知道相机外参，或者两个相机阵列的坐标系关系未知。

**优点**：
- 无需相机标定数据
- 自动计算最优变换
- 适用于大多数情况

**使用方法**：

```bash
python utils/covert/point_cloud_registration.py \
    --source_pcd /path/to/point_cloud_C.ply \
    --target_pcd /path/to/point_cloud_D.ply \
    --output_pcd /path/to/aligned_point_cloud_C.ply \
    --method icp \
    --voxel_size 0.05 \
    --max_iterations 30
```

**参数说明**：
- `--source_pcd`: 源点云文件路径（点云C）
- `--target_pcd`: 目标点云文件路径（点云D）
- `--output_pcd`: 输出对齐后的点云文件路径
- `--method`: 配准方法，使用 "icp"
- `--voxel_size`: 下采样体素大小（米），默认0.05。点云较大时可以增大此值以提高速度
- `--max_iterations`: ICP最大迭代次数，默认30

### 方法2：基于相机外参的配准

**适用场景**：已知两个相机阵列的标定数据，且两个阵列拍摄的是同一个物体。

**优点**：
- 如果相机标定准确，结果更可靠
- 计算速度快
- 支持单相机和多相机两种模式

#### 2.1 单相机方法（默认）

使用一个公共参考相机计算变换。

```bash
python utils/covert/point_cloud_registration.py \
    --source_pcd /path/to/point_cloud_C.ply \
    --target_pcd /path/to/point_cloud_D.ply \
    --output_pcd /path/to/aligned_point_cloud_C.ply \
    --method camera \
    --source_calibration /path/to/array_A_calibration.csv \
    --target_calibration /path/to/array_B_calibration.csv \
    --reference_camera Cam127
```

#### 2.2 多相机方法（推荐，更准确）

使用多个公共相机计算更准确的变换。工具会自动查找两个相机阵列中的公共相机，或您可以手动指定。

```bash
# 自动查找公共相机
python utils/covert/point_cloud_registration.py \
    --source_pcd /path/to/point_cloud_C.ply \
    --target_pcd /path/to/point_cloud_D.ply \
    --output_pcd /path/to/aligned_point_cloud_C.ply \
    --method camera \
    --source_calibration /path/to/array_A_calibration.csv \
    --target_calibration /path/to/array_B_calibration.csv \
    --use_multiple_cameras

# 手动指定公共相机
python utils/covert/point_cloud_registration.py \
    --source_pcd /path/to/point_cloud_C.ply \
    --target_pcd /path/to/point_cloud_D.ply \
    --output_pcd /path/to/aligned_point_cloud_C.ply \
    --method camera \
    --source_calibration /path/to/array_A_calibration.csv \
    --target_calibration /path/to/array_B_calibration.csv \
    --use_multiple_cameras \
    --common_cameras Cam127 Cam128 Cam126 Cam110
```

**参数说明**：
- `--source_calibration`: 相机阵列A的标定CSV文件路径
- `--target_calibration`: 相机阵列B的标定CSV文件路径
- `--reference_camera`: 用于对齐的参考相机名称（单相机方法），默认"Cam127"
- `--use_multiple_cameras`: 启用多相机方法（更准确）
- `--common_cameras`: 公共相机名称列表（多相机方法，可选，不指定则自动查找）

## Python代码中使用

```python
from pathlib import Path
from utils.covert.point_cloud_registration import register_and_align_point_clouds

# 方法1：使用ICP
transform_matrix, fitness = register_and_align_point_clouds(
    source_pcd_path=Path("/path/to/point_cloud_C.ply"),
    target_pcd_path=Path("/path/to/point_cloud_D.ply"),
    output_pcd_path=Path("/path/to/aligned_point_cloud_C.ply"),
    method="icp",
    voxel_size=0.05,
    max_iterations=30
)

# 方法2a：使用相机外参（单相机方法）
transform_matrix, _ = register_and_align_point_clouds(
    source_pcd_path=Path("/path/to/point_cloud_C.ply"),
    target_pcd_path=Path("/path/to/point_cloud_D.ply"),
    output_pcd_path=Path("/path/to/aligned_point_cloud_C.ply"),
    method="camera",
    source_calibration_csv=Path("/path/to/array_A_calibration.csv"),
    target_calibration_csv=Path("/path/to/array_B_calibration.csv"),
    reference_camera_name="Cam127"
)

# 方法2b：使用相机外参（多相机方法，推荐）
transform_matrix, _ = register_and_align_point_clouds(
    source_pcd_path=Path("/path/to/point_cloud_C.ply"),
    target_pcd_path=Path("/path/to/point_cloud_D.ply"),
    output_pcd_path=Path("/path/to/aligned_point_cloud_C.ply"),
    method="camera",
    source_calibration_csv=Path("/path/to/array_A_calibration.csv"),
    target_calibration_csv=Path("/path/to/array_B_calibration.csv"),
    use_multiple_cameras=True,
    common_camera_names=["Cam127", "Cam128", "Cam126"]  # 可选，None则自动查找
)
```

## 注意事项

1. **点云格式**：目前支持PLY格式的点云文件
2. **坐标系**：确保点云和相机标定使用相同的坐标系约定（本项目使用RDF坐标系）
3. **配准质量**：ICP方法会返回一个fitness分数（0-1），分数越高表示配准质量越好
4. **性能**：对于大型点云，建议使用较大的`voxel_size`进行下采样以提高速度

## 示例

假设您有以下文件：
- 相机阵列A的点云：`/data/array_A/point_cloud.ply`
- 相机阵列B的点云：`/data/array_B/point_cloud.ply`
- 相机阵列A的标定：`/data/array_A/calibration.csv`
- 相机阵列B的标定：`/data/array_B/calibration.csv`

要将阵列A的点云对齐到阵列B的位置：

```bash
# 使用ICP方法（推荐）
python utils/covert/point_cloud_registration.py \
    --source_pcd /data/array_A/point_cloud.ply \
    --target_pcd /data/array_B/point_cloud.ply \
    --output_pcd /data/array_A/aligned_point_cloud.ply \
    --method icp

# 或使用相机外参方法
python utils/covert/point_cloud_registration.py \
    --source_pcd /data/array_A/point_cloud.ply \
    --target_pcd /data/array_B/point_cloud.ply \
    --output_pcd /data/array_A/aligned_point_cloud.ply \
    --method camera \
    --source_calibration /data/array_A/calibration.csv \
    --target_calibration /data/array_B/calibration.csv
```

