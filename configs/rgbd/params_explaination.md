# FVO-GS-SLAM 模块开关、掩膜逻辑与深度策略整理

## 1. 模块开关含义

| 模块 | 决定的是 | 开关层级 |
|---|---|---|
| `use_fft_mask` | 高斯初始 `scale` 大小：高频区域更小，低频区域更大 | 总开关 |
| `use_freq_sampling_density` | 高斯初始采样密度：高频区域更密，低频区域更疏 | 依赖 `fft_mask` |
| `use_error_mask` | 哪里需要补点：空洞、深度穿透、RGB 错误 | 总开关 |
| `use_rgb_error_mask` | `error_mask` 中是否包含 RGB 颜色错误分量 | 依赖 `error_mask` |

## 2. Error Mask 组成

`error_mask` 由三个子掩膜组成：

```text
error_mask = alpha_mask | depth_error_mask | rgb_error_mask
```

| 子掩膜 | 条件 | 含义 |
|---|---|---|
| `alpha_mask` | `render_opacity < 0.98` | 地图覆盖不足 |
| `depth_error_mask` | `render_depth > gt_depth` 且误差 `> 10 × median` | 地图缺前景几何 |
| `rgb_error_mask` | `rgb_error > 0.5` | 颜色表达错误，需要单独开关控制 |

## 3. Surface Aware 相关开关组合

| `use_sa` | `use_sa_depth` | `use_sa_dist` | 效果 |
|---|---|---|---|
| `false` | `false` | `false` | 标准深度 + 无 `dist loss` |
| `false` | `false` | `true` | 标准深度 + 标准失真进入 loss |
| `true` | `false` | `false` | CUDA 计算 SA，但 loss 仍使用标准深度，无 dist |
| `true` | `false` | `true` | 标准深度 + SA variance 进入 loss |
| `true` | `true` | `false` | SA 深度进入 loss，dist 被 guard 跳过 |
| `true` | `true` | `true` | SA 深度 + SA variance 全部进入 loss，即完整 SA pipeline |

## 4. `use_sa_depth` 与 `depth_ratio` 的关系

| `use_sa_depth` | `depth_ratio` | `surf_depth`，即进入 loss 的深度 |
|---|---:|---|
| `true` | 任意值，被忽略 | SA expected depth |
| `false` | `0.0` | 纯 expected depth |
| `false` | `1.0` | 纯 median depth |
| `false` | `0.5` | expected depth 与 median depth 各占 50% |

## 5. 三种深度计算方式

| 深度类型 | 公式 | 离群 splat 影响 |
|---|---|---|
| 纯 expected depth | `Σ w_i · d_i / Σ w_i`，即加权平均 | 会被离群点拉偏 |
| 纯 median depth | 沿 ray 排序后取中值或主表面深度 | 基本不受离群点影响 |
| SA expected depth | `Σ w_i · d_i' / Σ w_i`，CUDA 将离群 splat 深度拉向 median 后再加权 | 部分抑制离群点影响 |

其中：

```text
d_i  = 原始 splat depth
d_i' = 经过 Surface aware adjustment 后的 splat depth
w_i  = alpha_i × transmittance_i
```

## 6. 数据集适用场景

### 6.1 Replica：无噪声或低噪声深度

Replica 深度较干净，初始点云通常紧贴真实几何表面。此时多数 splat 深度接近真值，expected depth 可以充分利用所有 splat 的加权信息，梯度连续且平滑，有利于 tracking 收敛。

因此，Replica 更适合：

```yaml
use_sa_depth: false
depth_ratio: 0.0
```

即优先使用纯 expected depth。

Replica 上不建议默认开启 SA expected depth 的原因是：SA expected 会将离群 splat 深度拉回 median，在桌角、墙角等锐利几何边界处可能造成过度平滑，进而损失几何精度，导致 tracking depth loss 的梯度偏移。

### 6.2 TUM：Kinect 深度噪声明显

TUM RGB-D 数据来自 Kinect，深度存在量化噪声、空洞和边缘飞点。部分 splat 初始位置可能偏离真实表面，纯 expected depth 容易被少数离群 splat 拉偏，导致 tracking 梯度方向错误。

因此，TUM 更适合：

```yaml
use_sa_depth: false
depth_ratio: 1.0
```

或者在完成充分验证后使用：

```yaml
use_sa_depth: true
```

也就是优先使用 median depth 或 SA expected depth。

## 7. 推荐配置

| 数据集 | 建议 `depth_ratio` | 建议深度策略 | 原因 |
|---|---:|---|---|
| Replica room0 | `0.0` | expected depth | 深度干净，expected depth 信息利用率高 |
| TUM fr1 / fr2 / fr3 | `1.0` | median depth | Kinect 深度噪声较明显，median depth 抗离群更稳 |

## 8. 实验解释要点

### 8.1 为什么 Replica 上 SA depth 可能变差

Replica 的深度干净，前景与背景边界清晰，expected depth 已经能给出平滑且准确的几何残差。SA expected depth 会把偏离 median 的 splat 深度向主表面拉回，在锐利边界处可能削弱前景与背景的深度差异，造成边界深度偏移。

这会使 tracking 阶段的 depth loss 对 pose 的梯度产生系统性偏差，最终导致 ATE 增大。

### 8.2 为什么 TUM 上 median 或 SA expected 可能更稳

TUM 深度含噪，边缘飞点和空洞较多，expected depth 容易被少数错误 splat 拉偏。median depth 天然抗离群；SA expected depth 是一种折中方案，它保留加权平均的连续梯度，同时抑制偏离主表面的深度贡献。

### 8.3 为什么 SA depth 和 SA dist 应该独立控制

SA depth 决定进入 depth loss 的渲染深度；SA dist 决定 mapping 阶段是否使用围绕 median depth 的深度方差正则。二者作用阶段和梯度来源不同：

```text
SA depth:
    主要影响 tracking / mapping 的 depth reconstruction loss。

SA dist:
    主要影响 mapping 阶段的几何压实和 floaters 抑制。
```

因此，二者不应强绑定。更合理的消融方式是：

```text
SA depth only
SA dist only
SA depth + SA dist
```

## 9. 一句话总结

expected depth 适合干净数据，因为信息利用率高、梯度平滑；median depth 和 SA expected depth 更适合噪声数据，因为抗离群能力更强。Replica 通常优先使用 expected depth，而 TUM 更适合 median depth 或经过验证后的 SA expected depth。