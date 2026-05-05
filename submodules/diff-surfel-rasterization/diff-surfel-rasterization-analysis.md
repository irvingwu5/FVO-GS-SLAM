# diff-surfel-rasterization 架构分析报告

> 基于源码完整阅读 + GitNexus 符号索引，分析日期：2026-05-06

---

## 1. 项目概述

**diff-surfel-rasterization** 是一个带相机位姿优化的可微 2D Gaussian Surfel 渲染器，基于原始 [2DGS diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization) 修改而来，用于 CVPR 2025 论文 *4DTAM: Non-rigid Tracking and Mapping via Dynamic Surface Gaussians*。

**核心增强**：在原始 2DGS 渲染器的基础上，增加了对相机位姿（SE(3)）的解析 Jacobian 计算，使渲染损失可以直接反向传播到相机外参 `w`（旋转增量）和 `trans`（平移增量），支持端到端的位姿优化。

**在原项目 FVO-GS-SLAM 中的角色**：作为 Gaussians → 图像的可微渲染后端，被 `gaussian_splatting/gaussian_renderer/__init__.py` 中的 `render()` 函数封装调用，同时支持 tracking 阶段的位姿精化和 mapping 阶段的场景优化。

---

## 2. 项目文件结构

```
diff-surfel-rasterization/
├── setup.py                              # PyTorch CUDAExtension 构建
├── CMakeLists.txt                        # 独立 CMake 构建（可选）
├── ext.cpp                               # PyBind11 模块绑定
├── rasterize_points.h                    # C 函数声明
├── rasterize_points.cu                   # CUDA 入口函数（封装层）
├── diff_surfel_rasterization/
│   └── __init__.py                       # Python 接口（nn.Module + autograd.Function）
├── cuda_rasterizer/
│   ├── rasterizer.h                      # Rasterizer 类声明
│   ├── rasterizer_impl.h                 # GeometryState/BinningState/ImageState 数据结构
│   ├── rasterizer_impl.cu                # 前向/反向主调度（preprocess → sort → render）
│   ├── forward.h / forward.cu            # 前向 CUDA kernel（preprocess + tile-based rendering）
│   ├── backward.h / backward.cu          # 反向 CUDA kernel（render backward + preprocess backward）
│   ├── config.h                          # 编译期常量（BLOCK_SIZE, NUM_CHANNELS）
│   ├── auxiliary.h                       # 辅助函数（坐标系变换、四元数、frustum culling）
│   └── math.h                            # Lie 代数（SO3/SE3 + mat33/mat34/mat44）
└── third_party/
    ├── glm/                              # GLM 数学库（CUDA 兼容子集）
    └── stbi_image_write.h                # 图像写入（调试用）
```

---

## 3. 模块清单与职责

### 3.1 Python 层（`diff_surfel_rasterization/__init__.py`）

| 类/函数 | 职责 |
|---------|------|
| `GaussianRasterizationSettings` | NamedTuple，封装渲染参数（图像尺寸、FOV、view/proj 矩阵、背景色、SH degree 等） |
| `GaussianRasterizer` | `nn.Module`，渲染器主类。持有 `raster_settings`，提供 `forward()` 和 `markVisible()` |
| `_RasterizeGaussians` | `torch.autograd.Function`，实现 `forward()` / `backward()` 静态方法，调用 C++/CUDA 扩展 |
| `rasterize_gaussians()` | 便捷函数，调用 `_RasterizeGaussians.apply(...)` |

### 3.2 C++/PyBind11 绑定层（`ext.cpp`）

向 Python 暴露 3 个 C 函数：
- `rasterize_gaussians` → `RasterizeGaussiansCUDA`
- `rasterize_gaussians_backward` → `RasterizeGaussiansBackwardCUDA`
- `mark_visible` → `markVisible`

### 3.3 CUDA 入口层（`rasterize_points.cu`）

| 函数 | 职责 |
|------|------|
| `RasterizeGaussiansCUDA` | 验证输入维度，分配输出 tensor，调用 `CudaRasterizer::Rasterizer::forward()` |
| `RasterizeGaussiansBackwardCUDA` | 验证输入，分配梯度 tensor，调用 `CudaRasterizer::Rasterizer::backward()` |
| `markVisible` | 调用 `CudaRasterizer::Rasterizer::markVisible()` |

### 3.4 CUDA 核心实现层（`cuda_rasterizer/`）

| 文件 | 关键函数/结构体 | 职责 |
|------|----------------|------|
| `rasterizer.h` | `Rasterizer` 类 | 静态方法接口：`forward()`, `backward()`, `markVisible()` |
| `rasterizer_impl.h` | `GeometryState`, `BinningState`, `ImageState` | GPU 内存管理结构体，含 `fromChunk()` 工厂方法 |
| `rasterizer_impl.cu` | `Rasterizer::forward()`, `Rasterizer::backward()`, `checkFrustum`, `duplicateWithKeys`, `identifyTileRanges` | 完整渲染管线调度 |
| `forward.cu` | `FORWARD::preprocess()`, `FORWARD::render()`, `preprocessCUDA`, `renderCUDA`, `compute_transmat`, `compute_aabb`, `computeColorFromSH` | 前向 CUDA kernel |
| `backward.cu` | `BACKWARD::preprocess()`, `BACKWARD::render()`, `renderCUDA`, `preprocessCUDA`, `compute_transmat_aabb`, `computeColorFromSH` | 反向 CUDA kernel |
| `config.h` | `NUM_CHANNELS=3`, `BLOCK_X=16`, `BLOCK_Y=16` | 编译期常量 |
| `auxiliary.h` | `in_frustum`, `transformPoint4x3/4x4`, `transformVec4x3`, `quat_to_rotmat`, `quat_to_rotmat_vjp`, `scale_to_mat`, `getRect`, `ndc2Pix` 等 | 设备端辅助函数 |
| `math.h` | `mat33`, `mat34`, `mat44`, `SO3`, `SE3` | Lie 代数（用于位姿 Jacobian 计算） |

---

## 4. 输入输出规格

### 4.1 Python `GaussianRasterizer.forward()` 输入

| 参数 | Shape | Dtype | 语义 |
|------|-------|-------|------|
| `means3D` | `(P, 3)` | float32 | Gaussian 中心世界坐标 (x, y, z) |
| `means2D` | `(P, 3)` | float32 | 屏幕空间点（占位，需 `requires_grad=True` 以获取 2D 梯度） |
| `opacities` | `(P, 1)` | float32 | 不透明度 α ∈ [0, 1] |
| `shs` | `(P, M, 3)` 或 `(0,)` | float32 | 球谐系数（M = (D+1)², D 为 SH degree） |
| `colors_precomp` | `(P, 3)` 或 `(0,)` | float32 | 预计算 RGB（与 SH 二选一） |
| `scales` | `(P, 2)` 或 `(0,)` | float32 | 2D Gaussian 缩放 (s_x, s_y) |
| `rotations` | `(P, 4)` 或 `(0,)` | float32 | 四元数旋转 (q_x, q_y, q_z, q_w) |
| `cov3D_precomp` | `(P, 9)` 或 `(0,)` | float32 | 预计算的变换矩阵 T（3×3 列主序），与 scales/rotations 二选一 |
| `w` | `(3,)` 或 `(0,)` | float32 | 相机旋转增量（轴角表示 ρ_1, ρ_2, ρ_3），用于位姿优化 |
| `trans` | `(3,)` 或 `(0,)` | float32 | 相机平移增量（θ_x, θ_y, θ_z），用于位姿优化 |

> P = Gaussian 数量, M = SH 系数数量

### 4.2 Python `GaussianRasterizer.forward()` 输出

| 返回值 | Shape | 语义 |
|--------|-------|------|
| `rendered_image` | `(3, H, W)` | 渲染 RGB 图像 |
| `radii` | `(P,)` int32 | 每个 Gaussian 的屏幕空间半径（0=不可见） |
| `allmap` | `(7, H, W)` | 多通道辅助输出（详见下表） |
| `n_touched` | `(P,)` int32 | 每个 Gaussian 被多少像素 T>0.5 时触及 |

### 4.3 `allmap` 7 通道布局

| 通道索引 | 常量名 | 内容 |
|---------|--------|------|
| `0` | `DEPTH_OFFSET` | 期望深度累积量（需除以 alpha） |
| `1` | `ALPHA_OFFSET` | 累积 alpha（1 - 透射率 T） |
| `2:5` | `NORMAL_OFFSET` | 累积视图空间法线 (nx, ny, nz) |
| `5` | `MIDDEPTH_OFFSET` | 中值深度（T 降至 0.5 时的深度） |
| `6` | `DISTORTION_OFFSET` | 深度失真累积量（用于 distortion loss） |

### 4.4 反向传播梯度

前向输出的 `rendered_image` 和 `allmap` 的梯度（`grad_out_color`, `grad_depth`），经反向传播后得到各输入参数的梯度：

| 梯度 | Shape | 对应参数 |
|------|-------|---------|
| `grad_means3D` | `(P, 3)` | means3D |
| `grad_means2D` | `(P, 3)` | means2D |
| `grad_sh` | `(P, M, 3)` | shs |
| `grad_colors_precomp` | `(P, 3)` | colors_precomp |
| `grad_opacities` | `(P, 1)` | opacities |
| `grad_scales` | `(P, 2)` | scales |
| `grad_rotations` | `(P, 4)` | rotations |
| `grad_cov3Ds_precomp` | `(P, 9)` | cov3Ds_precomp |
| `grad_w` | `(1, 3)` | w（旋转增量） |
| `grad_trans` | `(1, 3)` | trans（平移增量） |

---

## 5. 前向渲染流程

### 5.1 顶层调用链

```
Python: GaussianRasterizer.forward()
  → _RasterizeGaussians.apply()
    → _C.rasterize_gaussians()                    # PyBind11
      → RasterizeGaussiansCUDA()                  # rasterize_points.cu
        → CudaRasterizer::Rasterizer::forward()   # rasterizer_impl.cu
```

### 5.2 `Rasterizer::forward()` 六阶段流水线

```
阶段 1: preprocess     FORWARD::preprocess()      → preprocessCUDA kernel
  每个 Gaussian 独立:
  - in_frustum(): 视锥体裁剪
  - compute_transmat(): 计算 3×3 变换矩阵 T = M^T × world2ndc × ndc2pix
    （或将预计算的 cov3D_precomp 直接作为 T）
  - compute_aabb(): 计算 2D 屏幕空间包围盒 + 半径
  - computeColorFromSH(): SH → RGB 转换
  - 输出: radii[], means2D[], depths[], transMat[9P], rgb[3P], normal_opacity[4P]

阶段 2: prefix sum    cub::DeviceScan::InclusiveSum
  对 tiles_touched[] 做前缀和 → point_offsets[]
  总渲染实例数 num_rendered = point_offsets[P-1]

阶段 3: duplicate      duplicateWithKeys kernel
  每个可见 Gaussian 对其覆盖的每个 tile 生成一个 key-value pair:
  key = (tile_id << 32) | depth_as_uint32
  value = Gaussian index

阶段 4: radix sort     cub::DeviceRadixSort::SortPairs
  按 key 排序 → 同一 tile 内按深度排列

阶段 5: identify ranges  identifyTileRanges kernel
  从排序后的 key 列表中确定每个 tile 的 [start, end) 范围

阶段 6: render         FORWARD::render()           → renderCUDA kernel
  每个 tile 一个 block，block 内线程协作:
  - 逐 batch 加载 Gaussian 数据到 shared memory
  - 对每个像素计算 ray-splat 交点
  - α-blending（从前到后），早停（T < 0.0001）
  - 累积 RGB/深度/法线/distortion
  - 输出图像 + 辅助通道
```

### 5.3 核心数学：Ray-Splat Intersection（renderCUDA）

对每个 Gaussian，基于变换矩阵 T（3×3 列主序，用 `Tu, Tv, Tw` 表示三列）：

```
k = pix_x * Tw - Tu
l = pix_y * Tw - Tv
p = cross(k, l)              # 两个齐次平面的交线方向
s = (p_x/p_z, p_y/p_z)       # 交点参数
rho3d = ||s||²               # 3D 马氏距离
d = xy - pixf                # 到中心的像素偏移
rho2d = 2.0 * ||d||²         # 2D 低通滤波 (FilterInvSquare = 2.0)
rho = min(rho3d, rho2d)      # 取小（相当于对远处的 Gaussian 施加低通滤波）
depth = (s·Tw + Tw_z) if rho3d≤rho2d else Tw_z
alpha = min(0.99, opacity * exp(-0.5 * rho))
```

---

## 6. 反向传播流程

### 6.1 顶层调用链

```
Python: _RasterizeGaussians.backward()
  → _C.rasterize_gaussians_backward()              # PyBind11
    → RasterizeGaussiansBackwardCUDA()              # rasterize_points.cu
      → CudaRasterizer::Rasterizer::backward()      # rasterizer_impl.cu
```

### 6.2 `Rasterizer::backward()` 两阶段

```
阶段 1: BACKWARD::render()        → renderCUDA kernel
  重新遍历每个像素的所有贡献 Gaussian（从后往前）:
  - 重算 T 和 alpha（复现前向状态）
  - 计算 dL/d(alpha), dL/d(color), dL/d(depth), dL/d(normal)
  - 传播到 dL/d(transMat) 和 dL/d(means2D)（根据 rho3d≤rho2d 分支）
  - block reduction + atomicAdd 写入全局梯度缓冲

阶段 2: BACKWARD::preprocess()    → preprocessCUDA kernel
  对每个 Gaussian:
  - compute_transmat_aabb(): 反向传播 T 的梯度到:
    - scales, rotations（通过 quat_to_rotmat_vjp）
    - means3D
    - dL_dtau（6 维：3 平移 + 3 旋转，SE(3) 位姿梯度）
  - computeColorFromSH(): SH 梯度 → dL/d(sh), dL/d(means3D)
```

### 6.3 位姿 Jacobian 计算（核心创新）

在 `backward.cu` 的 `compute_transmat_aabb()` 中，T 的梯度通过链式法则传播到 view matrix：

```
dL/dT → dL/d(view_matrix_{3×4})     # T = M^T × projmat × ndc2pix
dL/d(view_matrix) → dL/d(ρ, θ)      # SE(3) 左乘扰动
```

具体实现（`math.h` 中的 `SE3` 类 + `backward.cu`）：
- 将 `dL/dView` 分解为对旋转（`θ`）和平移（`ρ`）的梯度
- 使用 `SO3::hat()` 的反对称矩阵、`SO3::Exp()` 的导数
- 最终输出 6 维 `dL_dtau = (dL/dρ_x, dL/dρ_y, dL/dρ_z, dL/dθ_x, dL/dθ_y, dL/dθ_z)`

在 Python 端（`__init__.py` backward），`dL_dtau` 被求和为 `grad_w`（旋转分量，shape (1,3)）和 `grad_trans`（平移分量，shape (1,3)）。

---

## 7. 模块间数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Python 层                                        │
│                                                                          │
│  GaussianModel           Camera                  render()               │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐          │
│  │ means3D     │     │ viewmatrix   │     │ raster_settings  │          │
│  │ opacities   │     │ projmatrix   │     │ bg_color         │          │
│  │ scales      │     │ cam_rot_delta│     │ scale_modifier   │          │
│  │ rotations   │     │ cam_trans_del│     │ image_height/W   │          │
│  │ shs         │     │ campos       │     └────────┬─────────┘          │
│  └──────┬──────┘     └──────┬───────┘            │                    │
│         │                   │                    │                    │
│         └───────────────────┼────────────────────┘                    │
│                             ▼                                          │
│              GaussianRasterizer.forward()                              │
│                             │                                          │
├─────────────────────────────┼──────────────────────────────────────────┤
│                         C++/PyBind11 层                                 │
│                             ▼                                          │
│                  _C.rasterize_gaussians()                               │
│                             │                                          │
├─────────────────────────────┼──────────────────────────────────────────┤
│                     CUDA 入口层 (rasterize_points.cu)                   │
│                             ▼                                          │
│              RasterizeGaussiansCUDA()                                   │
│                │                                                        │
│                ▼                                                        │
│     CudaRasterizer::Rasterizer::forward()                               │
│                │                                                        │
├────────────────┼────────────────────────────────────────────────────────┤
│  CUDA 核心层                                                           │
│                │                                                        │
│    ┌───────────▼──────────────┐                                        │
│    │ FORWARD::preprocess()    │  每 Gaussian: frustum cull,             │
│    │ (preprocessCUDA kernel)  │  compute transMat, compute AABB, SH→RGB│
│    └───────────┬──────────────┘                                        │
│                │ means2D, depths, transMat, rgb, normal_opacity         │
│    ┌───────────▼──────────────┐                                        │
│    │ cub::DeviceScan          │  前缀和 tiles_touched → offsets        │
│    └───────────┬──────────────┘                                        │
│    ┌───────────▼──────────────┐                                        │
│    │ duplicateWithKeys        │  key=(tile_id,depth), value=gaussian_id│
│    └───────────┬──────────────┘                                        │
│    ┌───────────▼──────────────┐                                        │
│    │ cub::DeviceRadixSort     │  按 tile_id+depth 排序                 │
│    └───────────┬──────────────┘                                        │
│    ┌───────────▼──────────────┐                                        │
│    │ identifyTileRanges       │  确定每个 tile 的 [start,end)         │
│    └───────────┬──────────────┘                                        │
│    ┌───────────▼──────────────┐                                        │
│    │ FORWARD::render()        │  Tile-based α-blending,               │
│    │ (renderCUDA kernel)      │  输出 RGB + allmap + n_touched         │
│    └──────────────────────────┘                                        │
│                                                                          │
│  === 反向传播时额外执行 ===                                              │
│                                                                          │
│    ┌──────────────────────────┐                                        │
│    │ BACKWARD::render()       │  逐 pixel 反向遍历 Gaussian,           │
│    │ (renderCUDA kernel)      │  dL→d(transMat), dL→d(means2D),       │
│    │                          │  dL→d(opacity), dL→d(color)            │
│    └───────────┬──────────────┘                                        │
│    ┌───────────▼──────────────┐                                        │
│    │ BACKWARD::preprocess()   │  transMat→scales/rots/means3D,        │
│    │ (preprocessCUDA kernel)  │  view→SE3 位姿梯度 dL_dtau            │
│    └──────────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 函数调用关系图

```
Python 侧:
  GaussianRasterizer.__init__(raster_settings)
  GaussianRasterizer.markVisible(positions) → _C.mark_visible()
  GaussianRasterizer.forward(means3D, means2D, opacities, shs, ...)
    └→ rasterize_gaussians(...)
         └→ _RasterizeGaussians.apply(...)
              ├─ forward():  _C.rasterize_gaussians(...)
              └─ backward(): _C.rasterize_gaussians_backward(...)

C++ 侧 (rasterize_points.cu):
  RasterizeGaussiansCUDA()
    ├─ CHECK_INPUT 验证
    ├─ torch::empty/zeros 分配输出
    └─ CudaRasterizer::Rasterizer::forward()
         ├─ GeometryState::fromChunk()
         ├─ ImageState::fromChunk()
         ├─ FORWARD::preprocess()    → preprocessCUDA kernel
         │    ├─ in_frustum()
         │    ├─ compute_transmat()
         │    │    ├─ quat_to_rotmat()
         │    │    └─ scale_to_mat()
         │    ├─ compute_aabb()
         │    ├─ computeColorFromSH()
         │    └─ getRect()
         ├─ cub::DeviceScan::InclusiveSum
         ├─ BinningState::fromChunk()
         ├─ duplicateWithKeys kernel
         │    └─ getRect()
         ├─ cub::DeviceRadixSort::SortPairs
         ├─ identifyTileRanges kernel
         └─ FORWARD::render()        → renderCUDA kernel
              ├─ cross()  (ray-splat intersection)
              └─ α-blending loop

  RasterizeGaussiansBackwardCUDA()
    ├─ CHECK_INPUT 验证
    ├─ torch::zeros 分配梯度
    └─ CudaRasterizer::Rasterizer::backward()
         ├─ GeometryState::fromChunk() (re-interpret forward buffers)
         ├─ BinningState::fromChunk()
         ├─ ImageState::fromChunk()
         ├─ BACKWARD::render()        → renderCUDA kernel
         │    ├─ α-blending 反传
         │    ├─ block_reduction (warp-level reduction)
         │    └─ atomicAdd → dL_d(transMat, means2D, opacity, color, normal)
         └─ BACKWARD::preprocess()    → preprocessCUDA kernel
              ├─ compute_transmat_aabb()
              │    ├─ quat_to_rotmat()
              │    ├─ quat_to_rotmat_vjp()     # 四元数 VJP
              │    ├─ transformVec4x3Transpose()
              │    ├─ SE3::R(), SE3::t()
              │    ├─ SO3::hat() → skew_symmetric
              │    └─ → dL_dtau[6]              # SE(3) 位姿梯度
              └─ computeColorFromSH() (backward)

  markVisible()
    └─ checkFrustum kernel
         └─ in_frustum()
```

---

## 9. Python 接口详解

### 9.1 `GaussianRasterizationSettings` (NamedTuple)

```python
GaussianRasterizationSettings(
    image_height: int,        # 输出图像高度
    image_width: int,         # 输出图像宽度
    tanfovx: float,           # tan(FoVx/2)
    tanfovy: float,           # tan(FoVy/2)
    bg: torch.Tensor,         # 背景色 (3,)
    scale_modifier: float,    # Gaussian 缩放修正系数
    viewmatrix: torch.Tensor, # 世界→视图 4×4 变换矩阵
    projmatrix: torch.Tensor, # 世界→NDC 4×4 全投影矩阵 (view @ proj)
    sh_degree: int,           # 球谐阶数 (0-3)
    campos: torch.Tensor,     # 相机世界坐标 (3,)
    prefiltered: bool,        # 是否预过滤（调试用）
    debug: bool,              # 调试模式（crash 时 dump 快照）
    cx: float,                # 主点 x 偏移（默认 0）
    cy: float,                # 主点 y 偏移（默认 0）
    projmatrix_raw: torch.Tensor,  # 原始投影矩阵（不含 view）
)
```

### 9.2 `GaussianRasterizer` (nn.Module)

```python
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings)
    def markVisible(self, positions: Tensor) -> Tensor  # → bool tensor (P,)
    def forward(self,
        means3D,         # (P, 3)
        means2D,         # (P, 3) 需 requires_grad=True
        opacities,       # (P, 1)
        shs=None,        # (P, M, 3) 或 None
        colors_precomp=None,  # (P, 3) 或 None（与 shs 二选一）
        scales=None,     # (P, 2) 或 None
        rotations=None,  # (P, 4) 或 None（与 cov3D_precomp 二选一）
        cov3D_precomp=None,  # (P, 9) 或 None
        w=None,          # (3,) 旋转增量
        trans=None,      # (3,) 平移增量
    ) → (image, radii, allmap, n_touched)
```

约束：
- `shs` 和 `colors_precomp` 必须且仅提供一个
- `(scales, rotations)` 和 `cov3D_precomp` 必须且仅提供一组
- `w` / `trans` 非 None 时启用位姿梯度

### 9.3 父项目中的集成方式

在 `gaussian_splatting/gaussian_renderer/__init__.py` 中：

```python
from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color, ...):
    # 1. 构建 raster_settings
    raster_settings = GaussianRasterizationSettings(
        image_height=..., image_width=...,
        tanfovx=..., tanfovy=...,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        ...
    )

    # 2. 创建 rasterizer
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 3. 调用渲染，传入位姿增量参数
    rendered_image, radii, allmap, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        w=viewpoint_camera.cam_rot_delta,        # ← 位姿增量
        trans=viewpoint_camera.cam_trans_delta,  # ← 位姿增量
    )

    # 4. 解析 allmap 多通道输出
    render_alpha = allmap[1:2]
    render_normal = allmap[2:5]
    render_depth_median = allmap[5:6]
    render_depth_expected = allmap[0:1] / render_alpha
    render_dist = allmap[6:7]
```

---

## 10. 关键技术细节

### 10.1 2D Gaussian 表示

每个 Gaussian 表示为一个 2D 平面椭圆（surfel），参数化方式：
- **位置**: 3D 世界坐标 `(x, y, z)`
- **旋转**: 四元数 `(q_x, q_y, q_z, q_w)`
- **缩放**: 2D 各向异性 `(s_x, s_y)`
- **变换矩阵 T**: 3×3 矩阵，将 tangent plane 坐标映射到图像平面

```
R = quat_to_rotmat(q)       # 四元数 → 3×3 旋转矩阵
S = diag(s_x, s_y, 1)       # 缩放矩阵
L = R × S                    # 局部切平面变换

M = [L[:,0]  0]              # 3×4: splat2world (tangent plane → world)
    [L[:,1]  0]
    [p_orig  1]

T = M^T × world2ndc × ndc2pix  # 3×3: tangent plane → pixels
```

### 10.2 低通滤波

为防止 2DGS 过小导致 aliasing，对屏幕空间半径施加下限：
- `FilterSize = sqrt(2)/2 ≈ 0.707`
- `FilterInvSquare = 2.0`
- 最终半径 = `max(max(extent_x, extent_y), cutoff * FilterSize)`
- 在 ray-splat 交会时，`rho = min(rho3d, rho2d)` 取较小者，相当于远距离自动施加低通

### 10.3 位姿 Jacobian 流

```
渲染损失 L 对像素颜色 C 的梯度
  ↓ (BACKWARD::render)
dL/dT  (对变换矩阵 T 的梯度)
  ↓ (compute_transmat_aabb)
dL/d(view_matrix)  →  dL/dρ (平移), dL/dθ (旋转)
  ↓ (SE3 代数)
dL_dtau = [dL/dρ, dL/dθ]  (6维)
  ↓ (Python backward 求和)
grad_w = Σ dL/dθ     (1×3)
grad_trans = Σ dL/dρ (1×3)
```

### 10.4 双面渲染 (DUAL_VISIBLE)

`auxiliary.h` 中定义了 `DUAL_VISIABLE 0`（默认关闭）、`BACKFACE_CULL 1`（默认开启）、`DETACH_WEIGHT 0`（默认不 detach）等编译开关，可通过修改后重新编译来控制渲染行为。

### 10.5 位姿增量参数化

`w` 和 `trans` 采用 SE(3) 对数映射参数化：
- `trans = (ρ_x, ρ_y, ρ_z)` — 平移向量
- `w = (θ_x, θ_y, θ_z)` — 旋转轴角
- 位姿更新：`T_new = Exp(τ) × T_old`，其中 `τ = (ρ, θ)`

这是标准的 on-manifold SE(3) 优化参数化。

---

## 11. 构建与依赖

### 构建方式 1：pip install（推荐）

```bash
pip install submodules/diff-surfel-rasterization/
```

`setup.py` 调用 `torch.utils.cpp_extension.CUDAExtension`，自动编译：
- `cuda_rasterizer/rasterizer_impl.cu`
- `cuda_rasterizer/forward.cu`
- `cuda_rasterizer/backward.cu`
- `rasterize_points.cu`
- `ext.cpp`

### 构建方式 2：CMake（独立调试）

```bash
cd submodules/diff-surfel-rasterization
mkdir build && cd build
cmake ..  # 需要 CUDA Toolkit
make
```

### 依赖

| 依赖 | 用途 |
|------|------|
| PyTorch (≥1.7) | Tensor + autograd + CUDAExtension |
| CUDA Toolkit (≥11.0) | CUDA 编译器 + cub 库 |
| GLM (header-only) | 矩阵/四元数运算（`third_party/glm/`） |
| CUB (header-only) | GPU radix sort + prefix sum（CUDA Toolkit 自带） |

---

## 12. 与原始 3DGS diff-gaussian-rasterization 的差异

| 特性 | 原始 3DGS | 本项目 2DGS |
|------|----------|------------|
| Gaussian 类型 | 3D 椭球体 | 2D 平面椭圆 (surfel) |
| 缩放参数 | 3 维 (s_x, s_y, s_z) | 2 维 (s_x, s_y) |
| 协方差 | 3×3 世界空间协方差 | 3×3 变换矩阵 T (tangent→pixels) |
| Ray 交会 | 2D 投影 + 2D Gaussian 评估 | 齐次平面交线 + 3D/2D 混合距离 |
| 深度计算 | 投影深度 | 交点参数 s 的加权深度 |
| 抗锯齿 | 低通滤波（EWA） | 低通滤波 + 2D/3D 混合距离 |
| 位姿 Jacobian | 无 | 有（SE(3) 解析梯度） |
| 法线输出 | 无 | 有（视图空间法线累积） |
| 失真输出 | 无 | 有（深度失真累积） |
| 中值深度 | 无 | 有（T 降至 0.5 时的深度） |
