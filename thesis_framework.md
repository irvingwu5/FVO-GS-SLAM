# 论文写作框架

## 论文题目与简称

> **FVO-GS-SLAM: Frequency-Aware 2D Gaussian SLAM with Submap Covisibility Handoff**
>
> **简称：FVO-GS-SLAM**（Fourier-Visual-Odometry Gaussian-Splatting SLAM）

---

## 论文主线

2D Gaussian Splatting 为 RGB-D SLAM 提供了显式、可微且具备较高渲染质量的场景表示，使相机位姿估计与场景表示优化能够通过渲染误差建立联系。然而，在以 Gaussian 地图渲染结果参与 tracking refinement 的在线系统中，位姿估计、地图优化和地图管理之间仍存在明显耦合。一方面，当场景存在弱纹理、运动模糊，或当前 Gaussian 地图尚未充分优化时，基于渲染误差的 photometric refinement 容易失去稳定约束，导致位姿估计漂移；另一方面，为控制显存占用和优化开销，Gaussian based SLAM 通常需要限制在线维护的 Gaussian 地图规模，但如果在子图切换时直接移除旧子图中仍与当前视角共视的稳定 Gaussian，新子图在初始化阶段将缺少可用于 tracking 和 rendering 的渲染支撑，从而破坏 tracking 连续性，并进一步放大位姿误差与地图退化之间的耦合。为此，本文提出 FVO-GS-SLAM：前端通过频域感知 Visual Odometry 提供相对独立的几何位姿初值，再利用 2DGS 可微渲染进行位姿精化；后端通过共视 Handoff、子图内 RSKM、轻量 Gaussian 评分和关键帧级回环闭合，在显存可控的前提下缓解子图切换造成的渲染支撑断裂，并提升 tracking 连续性、局部渲染质量和全局一致性。

---

## 摘要

建议结构（~250 词）：

1. **背景**（2-3 句）：2D Gaussian Splatting 作为显式、可微且高质量的场景表示的潜力；在以 Gaussian 地图渲染结果参与 tracking refinement 的在线系统中，弱纹理、运动模糊和地图未充分优化会削弱 photometric refinement 约束；为控制显存占用和优化开销而进行的子图切换如果采用硬切图，又会破坏 tracking 和 rendering 所需的渲染支撑。
2. **方法概述**（4-5 句）：提出 FVO-GS-SLAM，包含两大核心模块——（1）频域感知边缘VO与渲染精化：FFT 高通几何特征 + DT 对齐提供相对独立的几何位姿初值，再由 2DGS 可微渲染进行位姿精化；（2）共视引导的子图建图：运动自适应子图分解 + 共视 Handoff 在切图期间维持 tracking 和 rendering 的短期渲染支撑 + 子图内 RSKM 与关键帧级回环闭合提升局部渲染质量和全局一致性。
3. **关键结果**（2-3 句）：在 TUM RGBD、Replica、ScanNet++ 上的 ATE / PSNR / SSIM / FPS / 显存等关键指标。
4. **结论**（1 句）：本系统在 tracking 稳定性、子图切换期间渲染支撑连续性和局部渲染质量方面相比 baseline 的改进。

---

## 第一章：引言

### 1.1 研究背景与动机
- SLAM 在机器人、AR/VR 中的核心地位
- RGB-D SLAM 的演进：稀疏特征 → 稠密重建 → 神经隐式 → 3D Gaussian Splatting
- 3DGS 的优势：显式表示 + 可微渲染 + 实时高质量新视角合成

### 1.2 Gaussian based SLAM 与在线 2DGS SLAM 的核心挑战
- 引用综述论文的分析框架，聚焦两大根本问题：
  1. **渲染参与式 Tracking 在退化条件下易失稳**：在以 Gaussian 地图渲染结果参与 tracking refinement 的系统中，弱纹理、运动模糊和地图未充分优化会削弱 photometric refinement 约束，导致位姿漂移。
  2. **Gaussian 地图规模控制与硬切图破坏渲染支撑**：为控制显存占用和优化开销，Gaussian based SLAM 需要限制在线维护的 Gaussian 地图规模；但硬切图会移除仍与当前视角共视的稳定 Gaussian，使新子图缺少可用于 tracking 和 rendering 的渲染支撑。

### 1.3 本文的主要贡献
- 贡献一：频域感知边缘VO与渲染精化——频域高通几何特征 + DT 对齐提供相对独立的几何位姿初值，再结合 2DGS 可微渲染进行位姿精化
- 贡献二：共视引导的子图建图——共视 Handoff 维持子图切换期间的 tracking/rendering 渲染支撑 + 子图内 RSKM + 关键帧级回环闭合
- 贡献三：完整的 FVO-GS-SLAM 系统实现与多数据集验证

### 1.4 论文结构

---

## 第二章：相关工作

### 2.1 RGB-D SLAM
- 传统方法：KinectFusion, ElasticFusion, BundleFusion, ORB-SLAM3
- 学习式 SLAM：CNN-SLAM, DeepV2D, DROID-SLAM

### 2.2 3D Gaussian Splatting 与可微渲染
- 原始 3DGS 原理（Kerbl et al. 2023）
- 2DGS / Surfel 变体及在室内场景的优势

### 2.3 Gaussian based SLAM
- 先驱系统：MonoGS, SplaTAM, Gaussian-SLAM
- 子图化系统：LoopSplat（motion-based submap cutting）
- VO 辅助系统：EAGS-SLAM（Edge VO coarse-to-fine tracking）
- 各系统在 tracking 表示、mapping 表示、是否使用子图策略、是否依赖 render based refinement、loop closure 维度的对比

### 2.4 频域滤波与视觉特征
- FFT 在图像增强中的应用
- 高通滤波在 SLAM 中的角色（FGS-SLAM）

### 2.5 视觉位置识别与回环检测
- 传统方法：Bag-of-Words, NetVLAD
- 学习方法：CosPlace, MixVPR
- 大视觉模型：DUSt3R, MASt3R, Reloc3R

### 2.6 本章小结

---

## 第三章：方法

### 3.1 系统概述

#### 3.1.1 系统架构总览
- 三进程架构图：FrontEnd（主进程）→ BackEnd（独立进程）→ LoopClosureProcess（独立进程）
- 总体数据流：RGB-D 输入 → 跟踪 → 关键帧 → 子图建图 → 回环闭合 → 子图融合与全局评估

#### 3.1.2 坐标约定与位姿表示
- W2C / C2W 约定，SE(3) 参数化，子图间相对位姿与全局位姿的关系

#### 3.1.3 2D Gaussian 场景表示
- 参数模型（位置 / 旋转 / 缩放 / 不透明度 / 球谐系数 / 法线）
- Surfel 渲染原理，与 3DGS 的对比与选择理由

---

### 3.2 频域感知边缘视觉里程计与渲染精化跟踪

> **Frequency-Aware Edge VO with Render-Based Refinement**

#### 3.2.1 问题分析
- Render based photometric refinement 的固有问题（弱纹理 / 运动模糊 / 地图未成熟 / 位姿-地图耦合）
- 两级 tracking 的设计原则：几何初值先行，2DGS 渲染精化随后执行

#### 3.2.2 频域高通几何特征提取
- CLAHE 局部对比度增强
- FFT → Gaussian HPF → IFFT 频域处理
- Triangle 自适应阈值 mask 生成
- 不同场景下的定性分析

#### 3.2.3 稠密距离变换对齐
- 参考帧 DT 金字塔构建与 Sobel 梯度预计算
- cur→ref 投影与梯度查找
- 解析 SE(3) 雅可比推导（Kerl 2012）

#### 3.2.4 粗到精 LM 优化
- 阻尼 Gauss-Newton 更新
- 金字塔层级与迭代预算
- 收敛判据与质量评估（dt_mean / visible_ratio / iters）

#### 3.2.5 两级跟踪架构
- Frequency-Aware Edge VO 几何初值 → 2DGS 可微渲染精化（Adam on SE(3) delta, RGB L1 + DSSIM + depth L1）
- 参考帧自动刷新与质量监控

---

### 3.3 共视引导的子图建图

> **Covisibility-Guided Submap Mapping**

#### 3.3.1 问题分析
- 在线维护的 Gaussian 地图规模增长导致显存占用和优化开销上升
- 硬切图破坏 tracking 和 rendering 所需的渲染支撑，并导致新子图初始化不稳定
- 长期累积漂移缺乏全局修正
- 设计原则：tracking 连续性 + 局部子图独立性 + 显存可控 + 全局一致

#### 3.3.2 运动自适应子图分解
- 锚点相对运动阈值设计（平移 + 旋转）
- 切图触发与执行流程
- 全量 prune + optimizer state 重置，将当前参与优化的 Gaussian 范围限制在局部子图内
- 不同数据集的阈值适配

#### 3.3.3 共视感知 Gaussian Handoff
- Handoff 候选条件（seed frame + tail keyframes 共视，优先选择仍能支撑当前视角 tracking/rendering 的稳定 Gaussian）
- Frozen GaussianModel 设计与 optimizer-less 导出，用作短期只读渲染支撑
- 仅当前子图 coverage 补洞策略
- 自动退出机制（关键帧数 / 帧数 / 覆盖率）

#### 3.3.4 轻量几何评分增强
- 简单评分的局限性与 KNN 共享设计
- 六维几何特征：support / opacity / observation / area / normal consistency / local density
- 加权融合与 top-K 选择
- Fallback 安全机制

#### 3.3.5 子图内随机关键帧重放
- 全局 RSKM 的显存与遗忘问题
- 子图内 RSKM 的采样策略与强制当前帧比率

#### 3.3.6 关键帧级回环闭合
- CosPlace 视觉位置识别（ResNet18 + GeM）
- 学习式关键帧对粗位姿估计（Reloc3R）
- RGB-D 深度几何验证 + 对数空间尺度搜索（0.1-20×），三门验收
- 关键帧 Pose Graph 构建（temporal / handoff / loop 边）与 Open3D LM 优化
- Safety 评估（max correction / residual ratios / 鲁棒边剔除）
- 轨迹与 Gaussian 分层修正

---

## 第四章：实验评估

### 4.1 实验设置
- 数据集：TUM RGBD (fr1/desk, fr2/xyz, fr3/office), Replica (office0-4), ScanNet++
- 评估指标：ATE RMSE, PSNR, SSIM, LPIPS, FPS, GPU Memory Peak
- 硬件平台与软件环境
- Baseline：MonoGS, SplaTAM, LoopSplat, EAGS-SLAM, 自身消融变体

### 4.2 Tracking 精度评估
- 各数据集 ATE 对比表与轨迹可视化
- Tracking 配置消融（纯 photometric / +EdgeVO / 完整系统）

### 4.3 渲染质量评估
- 各数据集 PSNR / SSIM / LPIPS 对比表
- 渲染图像定性对比（选代表性帧）
- 子图融合前后渲染质量对比

### 4.4 子图与 Handoff 消融实验
- Handoff on/off 切图前后 tracking 连续性与渲染支撑对比
- RAP2DGS Lite on/off 评分效果
- RSKM on/off 渲染质量影响
- 子图统计、Gaussian 数量与显存分析

### 4.5 回环闭合评估
- LoopClosure mode 消融（off / detect_only / verify_only / keyframe_pgo）
- 回环检测成功/失败案例定性分析
- PGO 修正前后轨迹对比

### 4.6 运行时性能分析
- Tracking / Mapping / Loop Closure 各阶段耗时分解
- GPU 显存时间曲线与 Gaussian 数量增长曲线

### 4.7 鲁棒性分析
- 弱纹理场景表现
- 运动模糊帧 tracking 质量
- 大场景长时间运行稳定性

---

## 第五章：结论

### 5.1 本文工作总结
- 回顾两大创新及其解决的核心问题
- 总结各数据集关键实验结果
- 强调在 tracking 稳定性、子图切换渲染支撑连续性和局部渲染质量方面的贡献

### 5.2 未来工作展望
- 混合 Tracking 增强：引入 RGB-D ICP 作为 Edge VO fallback
- Map Confidence Tracking：per-pixel 置信度加权 tracking loss
- 动态场景处理：语义 mask + 几何一致性动态检测与降权
- 高斯压缩：voxel anchoring / vector quantization 降低存储
- IMU 融合：提升快速运动和模糊场景鲁棒性
- 终身 SLAM：archived 子图的终身维护与增量更新

---

## 致谢

---

## 参考文献

---

## 附录

### A. 配置参数完整列表
### B. 数据集预处理细节
### C. 补充实验结果
### D. 符号表
