# FFT Edge VO 消融实验报告

**实验日期**: 2026-05-06
**测试序列**: TUM fr3_long_office_household (2515 帧)
**回环检测**: 关闭 (use_loop_closure=false，该序列无回环)
**渲染评估**: 开启 (eval_rendering=true)

---

## 1. 实验设计

消融两个独立因子：

| 因子 | 开关字段 | 作用 |
|---|---|---|
| **VO Prior** | `FFTEdgeVO.use_fft_edge_vo` | FFT mask DT 对齐 + Gauss-Newton LM，为可微渲染精化提供初始位姿 |
| **FFT Mask** | `Ablation.use_fft_mask` | 根据纹理频率控制新 Gaussian 初始 scale（高频→小尺度，低频→大尺度） |

四个实验：

| 实验 | VO Prior | FFT Mask | 配置文件 |
|---|---|---|---|
| A0 (baseline) | ✓ | ✓ | `ablation_fft_vo/A0_baseline.yaml` |
| A1 | ✗ | ✓ | `ablation_fft_vo/A1_no_vo.yaml` |
| A2 | ✓ | ✗ | `ablation_fft_vo/A2_no_fft_mask.yaml` |
| A3 | ✗ | ✗ | `ablation_fft_vo/A3_no_fft_all.yaml` |

---

## 2. 实验结果

### 2.1 核心指标汇总

| 实验 | VO Prior | FFT Mask | ATE RMSE (cm) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Depth L1 (cm) ↓ | 全局高斯数 | 子图数 |
|---|---|---|---|---|---|---|---|---|---|
| A0 (baseline) | ✓ | ✓ | **2.306** | 24.93 | 0.8408 | 0.2032 | 16.03 | 125,560 | 8 |
| A1 | ✗ | ✓ | 2.432 | 24.93 | 0.8405 | 0.2011 | 15.89 | 122,741 | 8 |
| A2 | ✓ | ✗ | 2.355 | **25.14** | **0.8426** | **0.2007** | 16.90 | 131,595 | 8 |
| A3 | ✗ | ✗ | 2.828 | 25.00 | 0.8415 | 0.2016 | **15.74** | 132,539 | 8 |

### 2.2 相对变化（以 A0 为基准）

| 实验 | ATE Δ | PSNR Δ | SSIM Δ | LPIPS Δ | Depth L1 Δ | 高斯数 Δ |
|---|---|---|---|---|---|---|
| A1 (no VO) | **+5.5%** | +0.0% | -0.03% | -1.0% | -0.9% | -2.2% |
| A2 (no FFT mask) | +2.1% | +0.8% | +0.2% | -1.2% | +5.4% | +4.8% |
| A3 (all off) | **+22.6%** | +0.3% | +0.08% | -0.8% | -1.8% | +5.6% |

### 2.3 ATE 演化曲线分析

每段 ATE RMSE 随时间单调递增（符合开环里程计漂移特征）。关键观察：

- **A0 与 A1 的 ATE 差距随序列增长逐渐拉大**：前半段（帧 0~1200）两者几乎重合，后半段 A1 开始落后，最终差距 0.13 cm
- **A2 的 ATE 早期（帧 500~1200）明显差于 A0**：中期 A2 的 ATE 一度达到 2.23 cm（A0 同期约 1.39 cm），后期差距缩小
- **A3 的 ATE 全程持续恶化**：最终 ATE 为 2.83 cm，比 A0 高 22.6%

---

## 3. 分析

### 3.1 VO Prior 的作用

**结论：提供有意义的但非关键的 tracking 改进。**

- VO 关闭后 ATE 退化 5.5%（2.306→2.432 cm），说明可微渲染精化有能力从"上一帧位姿"初值恢复出较好的位姿
- 差距主要在后半段累积，说明 VO 初值在长序列中更有价值：当累计漂移增大时，恒定速度假设更容易失效
- **渲染质量完全不受影响**（PSNR/SSIM/LPIPS 与 A0 相同），说明 VO 不直接参与建图，仅影响位姿精度
- VO 的计算开销极低：参考帧 DT 只算一次、后续帧复用，每帧仅做粗到精 LM 优化（通常 3-8 次迭代）

**机制解释**：VO 提供基于几何对齐的位姿初值，相比"上一帧位姿"的恒定速度假设，能更好地处理加速度和旋转变化。render refinement 虽然有 100-120 次迭代的优化能力，但其收敛 basin 有限——当初值离真值太远时，基于渲染的梯度优化可能陷入局部极小值。

### 3.2 FFT Mask 的作用

**结论：对渲染质量影响微弱，主要价值在于控制 Gaussian 数量。**

- 关闭 FFT mask 后 ATE 仅退化 2.1%（2.306→2.355 cm），影响很小
- **PSNR 反而提升 0.21 dB**（24.93→25.14），SSIM 略升，Depth L1 变差 5.4%
- **Gaussian 数量增加 4.8%**（125,560→131,595）：没有 freq-guided scale 初始化，低频纹理区用更小的初始 scale，需要更多 Gaussian 覆盖
- 渲染质量和 ATE 几乎没有退化，说明 Gaussian 优化过程（densify/prune/opacity reset）能有效补偿初始 scale 的差异

**机制解释**：`use_fft_mask=true` 时，低频区域（白墙等）的初始 Gaussian scale 乘以 `low_freq_scale_multiplier=2.0`，用更大的 Gaussian 覆盖平坦区域。关闭后所有区域用统一 scale，低频区需要更多小 Gaussian 拼凑覆盖。虽然最终 PSNR 差异不大（优化补偿），但更多 Gaussian 意味着更高的存储和渲染开销。

### 3.3 两因子交互效应

A3（全关）的 ATE 退化（+22.6%）远超两因子独立效应之和（~7.6%），说明存在**正向协同效应**：

- VO 提供更好的位姿初值 → 渲染精化从更好的起点开始 → Gaussian 优化在更准确的相机位姿下进行 → 更准确的 Gaussian 又反过来改善下一帧的 tracking
- FFT mask 提供更好的初始 scale → Gaussian 覆盖更高效 → 渲染质量更稳定 → tracking loss 的梯度更可靠
- 两者同时缺失时，误差通过 tracking→mapping→tracking 循环放大

---

## 4. 结论与建议

### 是否应移除 FFT Edge VO？

**建议：保留。**

| 维度 | 评估 |
|---|---|
| ATE 收益 | +5.5%，有意义但不关键 |
| 渲染收益 | 无直接影响 |
| 计算开销 | 极低（DT 复用，3-8 次 LM 迭代/帧） |
| 代码复杂度 | 中等（~500 行独立模块，接口清晰） |
| 鲁棒性价值 | 加速度/旋转较大时恒定速度假设失效，VO 提供安全的 fallback |

VO 以极低的计算代价提供 5.5% 的 ATE 改进，且代码已稳定、接口清晰。没有充分的理由移除。

### 是否应移除 FFT Mask？

**建议：保留，但可考虑调整参数。**

| 维度 | 评估 |
|---|---|
| ATE 收益 | +2.1%，很小 |
| 渲染收益 | PSNR 略降 0.21 dB（但深度 L1 改善 5.4%） |
| Gaussian 效率 | 减少 4.8% 点数 |
| 计算开销 | FFT+IFFT+Triangle 阈值，中等 |

FFT mask 的核心价值不在渲染质量（均匀初始 scale 也能达到类似 PSNR），而在 **Gaussian 效率**——用更少的点覆盖低频区域。当前 `low_freq_scale_multiplier=2.0` 参数可能偏保守，可尝试加大到 3.0-5.0 以进一步减少点数。

### 最终建议

**两个都保留**。VO Prior 以极低成本提供 ATE 改进，FFT Mask 以中等成本提供 Gaussian 效率改进。两者协同作用显著（A3 退化 22.6%），说明它们在系统中扮演互补角色。

---

## 5. 论文消融表（LaTeX）

```latex
\begin{table}[t]
\centering
\caption{Ablation of FFT Edge VO components on TUM fr3\_long\_office\_household.
Loop closure is disabled for all experiments to isolate the VO effect.}
\label{tab:fft_vo_ablation}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccrrrrc}
\toprule
& \textbf{VO Prior} & \textbf{FFT Mask} & \textbf{ATE RMSE} & \textbf{PSNR} & \textbf{SSIM} & \textbf{LPIPS} & \textbf{Depth L1} & \textbf{\#Gauss.} \\
& & & (cm) $\downarrow$ & (dB) $\uparrow$ & $\uparrow$ & $\downarrow$ & (cm) $\downarrow$ & \\
\midrule
A0 & \checkmark & \checkmark & 2.306 & 24.93 & 0.8408 & 0.2032 & 16.03 & 125,560 \\
A1 & $\times$   & \checkmark & 2.432 & 24.93 & 0.8405 & 0.2011 & 15.89 & 122,741 \\
A2 & \checkmark & $\times$   & 2.355 & 25.14 & 0.8426 & 0.2007 & 16.90 & 131,595 \\
A3 & $\times$   & $\times$   & 2.828 & 25.00 & 0.8415 & 0.2016 & 15.74 & 132,539 \\
\bottomrule
\end{tabular}
\end{table}
```

### 论文讨论段落（模板）

> **FFT Edge VO ablation.** Table~\ref{tab:fft_vo_ablation} reports the ablation of two components in our FFT Edge VO module: the VO pose prior and the FFT-guided Gaussian scale initialization. Removing the VO pose prior (A1) increases ATE RMSE by 5.5\% (2.306→2.432~cm), confirming that the DT-based geometric alignment provides meaningful initialization beyond a constant-velocity prior, though the differentiable render refinement is largely capable of recovering from the simpler initialization. Disabling FFT-guided scale initialization (A2) yields comparable ATE (+2.1\%) and slightly higher PSNR (+0.21~dB), at the cost of 4.8\% more Gaussians. This indicates that the primary value of the FFT mask lies in Gaussian efficiency — using larger initial scales in textureless regions reduces the total primitive count — rather than in rendering fidelity, as the subsequent densification and pruning operations can largely compensate for uniform initialization. Removing both components (A3) degrades ATE by 22.6\%, revealing a synergistic interaction: better pose initialization improves the mapping quality, which in turn provides more reliable tracking gradients. We retain both components in the final system given their low computational overhead and complementary benefits.

---

## 6. 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/rgbd/ablation_fft_vo/A0_baseline.yaml --eval
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/rgbd/ablation_fft_vo/A1_no_vo.yaml --eval
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/rgbd/ablation_fft_vo/A2_no_fft_mask.yaml --eval
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/rgbd/ablation_fft_vo/A3_no_fft_all.yaml --eval
```

---

## 7. 原始日志数据

| 实验 | 最终 ATE RMSE (m) | PSNR (dB) | SSIM | LPIPS | Depth L1 (m) | 全局高斯数 | 子图数 |
|---|---|---|---|---|---|---|---|
| A0 | 0.023060 | 24.9303 | 0.8408 | 0.2032 | 0.1603 | 125,560 | 8 |
| A1 | 0.024325 | 24.9314 | 0.8405 | 0.2011 | 0.1589 | 122,741 | 8 |
| A2 | 0.023546 | 25.1415 | 0.8426 | 0.2007 | 0.1690 | 131,595 | 8 |
| A3 | 0.028282 | 24.9974 | 0.8415 | 0.2016 | 0.1574 | 132,539 | 8 |

日志路径: `/home/wxy/Downloads/fft_ablation_log/`

---

## 8. fr3_office 序列 2000 帧后轨迹漂移分析

### 8.1 漂移现象

四个实验均在最后子图（submap 7，帧 ~1973→2515）出现 ATE 加速增长。以 A0 为例，将 ATE 演化分为三段：

| 阶段 | 帧范围 | ATE 范围 (m) | 漂移速率 (m/frame) |
|---|---|---|---|
| 早期 | 0 ~ 400 | 0.005 → 0.012 | ~1.8×10⁻⁵ |
| **中期（平坦期）** | **400 ~ 1950** | **~0.012（几乎不变）** | **~0** |
| 末期（submap 7） | 1950 → 2515 | 0.015 → 0.023 | **~1.6×10⁻⁵** |

中期约 1550 帧内 ATE 几乎零增长，说明系统在该段跟踪非常精确。漂移集中在最后一个子图。

四个实验的 submap 7 漂移对比：

| 实验 | cut 帧 | cut 时 ATE (cm) | 最终 ATE (cm) | ΔATE (cm) | 漂移速率 (m/fr) |
|---|---|---|---|---|---|
| A0 (baseline) | 1973 | 1.515 | 2.306 | 0.791 | 1.57×10⁻⁵ |
| A1 (no VO) | 1975 | 1.299 | 2.432 | 1.133 | 2.17×10⁻⁵ |
| A2 (no FFT mask) | 1975 | 1.874 | 2.355 | 0.481 | 0.95×10⁻⁵ |
| A3 (all off) | 1976 | 1.680 | 2.828 | 1.148 | 2.22×10⁻⁵ |

**关键发现**：A2（无 FFT mask）虽然 cut 时 ATE 更高，但 submap 7 内的漂移速率最小（0.95×10⁻⁵）。A1/A3 漂移速率最大。VO 的缺失放大了 submap 7 的漂移（A1 比 A0 快 38%），FFT mask 的影响则复杂（A2 漂移速率最低但起始 ATE 高）。

### 8.2 根因：Handoff 覆盖缺口导致的子图冷启动退化

漂移与 submap 7 的启动过程高度同步。子图切换的详细时间线（以 A0 为例）：

```
Frame 1973: Submap 7 启动
           ├─ 旧 submap 6 冻结，11040 个 Gaussians 参与 handoff 评分
           ├─ RAP2DGS Lite 选择 1802 个边界 Gaussian (16.3%)
           └─ 新 submap 从 seed frame 播种初始 Gaussian

Frame 1978 (age=5):  active_cov = 0.478  ← 仅 47.8% 像素由新子图覆盖
Frame 1983 (age=10): active_cov = 0.689  ← 68.9% 覆盖
Frame ~1988:         Handoff 关闭（满 3 个关键帧）
                     ← 31.1% 像素失去渲染支持！

Frame 1988~2050:     覆盖恢复期
                     ├─ Error mask 引导播种填补空洞
                     ├─ 新 Gaussian 初始 opacity=0.01, 需多轮 mapping 才能生效
                     └─ Tracking 梯度噪声增大，位姿优化质量下降

Frame 2050~2515:     误差传播期
                     └─ 恢复期内引入的位姿误差被固化为关键帧位姿，后续持续累积
```

**核心问题**：Handoff 过早退出（仅 3 个关键帧 / ~15 帧），退出时 active coverage 仅 68.9%。这意味着：

1. **渲染空洞**：31% 的像素在 handoff 退出后失去有效的 Gaussian 覆盖
2. **Tracking 退化**：空洞区域的 tracking loss 梯度不可靠，位姿优化收敛到次优解
3. **误差固化**：次优的位姿估计被插入为关键帧，其位姿成为后端建图的"真值"
4. **无法修正**：作为最后一个子图，没有后续子图切图来重置累积误差

### 8.3 为什么中期子图没有这个问题？

中期子图（submap 1-6）同样经历 handoff 冷启动，但它们：

1. **起始误差更小**：前一个子图的 ATE 累积还很低，seed frame 位姿更准
2. **有后续子图"兜底"**：即使引入了小误差，下一个子图切图时误差被"吸收"进新子图的参考系，不会在全局 ATE 中持续累积
3. **最后一个子图承担了所有未修正误差**：submap 7 是终点，前面所有子图切图引入的微小不连续性最终都体现在 submap 7 的轨迹末端

这解释了为什么 ATE 曲线在中期完全平坦（误差被逐个 submap 吸收），而在最后 500 帧快速上升（无处可逃）。

### 8.4 为什么 VO 的缺失会放大漂移？（A1 vs A0）

VO 提供基于几何对齐的位姿初值。在 handoff 覆盖恢复期内：
- **有 VO（A0）**：即使渲染空洞多，VO 的 DT 对齐仍能提供合理的位姿初值，render refinement 只需微调
- **无 VO（A1）**：完全依赖渲染梯度，空洞区域梯度噪声直接导致位姿估计恶化

A1 比 A0 的 submap 7 漂移速率高 38%（2.17 vs 1.57×10⁻⁵），直接体现了 VO 在稀疏覆盖场景下的鲁棒性价值。

### 8.5 改进建议

| 优先级 | 方案 | 预期效果 |
|---|---|---|
| **高** | 提高 `handoff_warmup_keyframes` 从 3 到 5-8 | 延长 handoff 保护期，让 active coverage 达到 >85% 再退出 |
| **高** | 添加 `handoff_min_coverage` 退出条件（如 0.85） | 覆盖不足时即使关键帧数达标也不退出 |
| 中 | 提高 `handoff_warmup_frames` 从 20 到 40 | 给最后一个子图更多暖启动帧数 |
| 中 | 最后一个子图不减 handoff，保持到序列结束 | 避免"最后子图承担全部误差"问题 |
| 低 | 实现 handoff 渐进衰减（gradual decay）代替硬切换 | 平滑过渡，减少渲染突变
