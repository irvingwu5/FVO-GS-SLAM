# FVO-GS-SLAM 系统架构文档

> **FVO-GS-SLAM: Frequency-Aware 2D Gaussian SLAM with Submap Covisibility Handoff**
>
> 基于 RGBD + 2D Gaussian Splatting 的静态室内 SLAM 系统，目标是在稳定轨迹精度的前提下提升渲染质量。三个并行进程 + 一个主进程后处理阶段，围绕两大核心问题展开：
>
> - **问题一：在线 Tracking 精度易受退化条件影响** — 弱纹理/模糊下 photometric 约束失效 → 频域感知边缘VO与渲染精化解决
> - **问题二：Gaussian 膨胀与硬切图导致渲染质量下降** — 显存不可控/切图渲染断裂/新子图初始化不稳定 → 共视引导的子图建图（Handoff + RSKM + 回环闭合）

```text
RGBD 序列 → FrontEnd (主进程) → BackEnd (独立进程) → LoopClosureProcess (独立进程) → 主进程融合评估
```

---

## 一、模块清单与职责

### 1. 入口与调度

| 文件 | 职责 |
|---|---|
| `slam.py` | 主入口，创建 GaussianModel / Dataset / FrontEnd / BackEnd / LoopClosure，启动多进程，前端跑完后停止后端和回环，流式加载子图 ckpt、应用 PGO 修正、融合全局 Gaussian、评估 ATE + 渲染质量 |

### 2. 频域感知边缘VO与渲染精化跟踪

> **Frequency-Aware Edge VO with Render-Based Refinement**

| 文件 | 职责 |
|---|---|
| `utils/slam_frontend.py` | 核心前端。逐帧：FFT mask → Edge VO 几何初值 → 可微渲染精化 → 关键帧决策 → 子图运动监控/切图 → 队列通信 |
| `utils/fft_edge_vo.py` | 频域感知 Edge VO。cur→ref 方向：当前帧 3D 点投影到参考帧 DT 金字塔，解析 SE(3) 雅可比 + 阻尼 Gauss-Newton（LM 风格），粗到精金字塔优化 |
| `utils/fft_filter.py` | FFT 高通 mask：CLAHE → FFT → Gaussian HPF → IFFT → Triangle 阈值 → bool mask。被 Edge VO（几何特征筛选）和 Gaussian 播种（初始 scale 控制）共用 |

### 3. 后端 Mapping（独立进程）

| 文件 | 职责 |
|---|---|
| `utils/slam_backend.py` | 后端建图。seed 帧初始化 → 关键帧扩展 Gaussian → RGB+depth+normal(FDN) 联合优化 → densify/prune/opacity reset → occ_aware_visibility → RSKM 随机关键帧重放 → Handoff 边界高斯选择 → 保存子图 ckpt |
| `gaussian_splatting/scene/gaussian_model.py` | 2DGS 参数模型：`_xyz`, `_features_dc`, `_opacity`, `_scaling`, `_rotation`, `_normal`。核心方法：`extend_from_pcd_seq()` 扩展点云、`densify_and_prune()` 增删高斯、`prune_points()` 剪枝、`capture_dict()` 导出、`training_setup()` 优化器初始化 |
| `gaussian_splatting/gaussian_renderer/__init__.py` | 可微 2DGS surfel 渲染：输出 RGB、depth、opacity、normal、visibility、radii、n_touched |
| `utils/slam_utils.py` | Tracking loss（RGB L1+DSSIM+depth L1）和 Mapping loss（同上 + FDN normal），以及图像梯度工具函数 |

### 4. 共视引导的子图建图

> **Covisibility-Guided Submap Mapping**

| 文件 | 职责 |
|---|---|
| `utils/slam_frontend.py` (子图部分) | `compute_submap_motion()` 计算当前帧相对于子图锚点的平移/旋转；`should_start_new_submap()` 阈值判断；`perform_submap_cut()` 触发切图 |
| `utils/slam_backend.py` (Handoff 部分) | 切图时根据 seed 帧 + 尾部关键帧共视关系选择旧子图边界 Gaussian → 导出为 frozen GaussianModel（无 optimizer）→ 发送给前端用作短期 tracking 支撑 |
| `utils/rap2dgs_lite/scorer.py` | RAP2DGS Lite 评分器：在 candidate_mask 内通过单次共享 KNN 计算 6 维几何特征（support/opacity/observation/area/normal consistency/local density），加权融合 |
| `utils/rap2dgs_lite/selector.py` | Top-K 选择 + fallback 逻辑 |
| `utils/rap2dgs_lite/feature_utils.py` | 归一化 / sanitize / chunked KNN / normal consistency 工具函数 |
| `utils/loop_closure.py` | LoopClosureProcess 主控。CosPlace 描述子提取 → 关键帧级检索 → Reloc3R 粗位姿 → 深度验证 → PGO trial → 修正写入 |
| `utils/reloc3r_adapter.py` | Reloc3R（DUSt3R 变体）关键帧对粗位姿估计，保留原始尺度 |
| `utils/loop_depth_verifier.py` | RGB-D 深度几何验证：对数空间尺度搜索（0.1-20×），三门验收 |
| `utils/keyframe_pgo.py` | 关键帧级 PGO：图构建（temporal/handoff/loop），Open3D LM 优化，safety 评估，分层修正 |

### 5. 工具与辅助

| 文件 | 职责 |
|---|---|
| `utils/camera_utils.py` | Camera 类（`viewpoint.T` = 全局 W2C，`inv(T)` = 全局 C2W） |
| `utils/dataset.py` | RGBD 数据集加载（TUM / Replica / ScanNet++ / Realsense） |
| `utils/pose_utils.py` | SE(3) 位姿更新工具 |
| `utils/eval_utils.py` | ATE 轨迹评估 + 渲染质量评估（PSNR/SSIM）+ PLY 保存 |
| `utils/config_utils.py` | YAML 配置加载 |
| `utils/logging_utils.py` | 日志工具 |
| `utils/multiprocessing_utils.py` | FakeQueue（单线程模式用） |
| `utils/normal_utils.py` | 法线计算工具 |
| `utils/point_utils.py` | 点云工具 |
| `utils/normal_mask_utils.py` | 法线 mask 工具 |
| `utils/ray_cache.py` | 射线缓存 |
| `utils/gaussian_state_manager.py` | Gaussian 状态管理 |

---

## 二、数据流

### 2.1 前端 → 后端（Queue 消息）

```
FrontEnd → backend_queue → BackEnd
  ["init",        frame_idx, viewpoint, depth_map]        # 初始化首个关键帧
  ["keyframe",    frame_idx, viewpoint, window, depth_map] # 每个关键帧
  ["new_submap",  submap_id, relative_pose, seed_c2w]      # 切图通知
  ["pause"]                                                # 暂停后端
  ["stop"]                                                 # 终止后端
```

### 2.2 后端 → 前端（Queue 消息）

```
BackEnd → frontend_queue → FrontEnd
  ["init",         gaussians, visibility, keyframes, handoff_data]
  ["keyframe",     gaussians, visibility, keyframes, handoff_data]
  ["sync_backend", gaussians, visibility, keyframes, handoff_data]
```

其中 `handoff_data` = `(frozen_gaussian_model, age_frames, warmup_frames)` 或 `None`。

### 2.3 后端 → 回环检测（Queue 消息）

```
BackEnd → loop_queue → LoopClosureProcess
  ["submap_saved", submap_id, ckpt_path, kf_image_paths, kf_depth_paths]
  ["stop"]
```

### 2.4 主进程融合流

```
主进程（slam.py）停止后端和回环后：
  1. 从磁盘 glob 子图 ckpt
  2. 读取 correct_tsfm（旧 PGO）或 keyframe_pgo_result.json（新 PGO）
  3. rigid_transform_2dgs 应用修正
  4. torch.cat 拼接所有子图 Gaussian 参数
  5. 评估 ATE + 渲染质量
```

---

## 三、单帧 Pipeline（FrontEnd 主循环）

```text
Frame i
  │
  ├─ 1. 构建 Camera 对象（RGB + depth）
  ├─ 2. 生成 FFT 高通 mask（fft_filter.py）
  ├─ 3. FFTEdgeVO.track() → 稠密 DT 对齐初值
  │     ├─ 当前帧 3D 点反投影
  │     ├─ 投影到参考帧 DT 金字塔 + Sobel 梯度
  │     └─ 粗到精 LM 优化（解析雅可比）
  ├─ 4. 可微渲染位姿精化（Adam 优化 cam_rot_delta / cam_trans_delta）
  │     用 RGB L1 + DSSIM + depth L1 tracking loss
  ├─ 5. 自动刷新 FFTEdgeVO reference（质量下降时）
  ├─ 6. 关键帧决策（overlap ratio + translation 阈值）
  │     ├─ 是 → 发送 ["keyframe", ...] 到后端
  │     └─ 接收后端 ["sync_backend", ...] 更新 Gaussian 快照
  ├─ 7. 子图运动监控
  │     ├─ compute_submap_motion() 计算相对锚点运动
  │     └─ should_start_new_submap() → perform_submap_cut()
  │        发送 ["new_submap", ...] 到后端
  └─ 8. 更新 occ_aware_visibility + 滑动窗口
```

---

## 四、回环检测 Pipeline（LoopClosureProcess）

```text
Stage 0: 模式控制（off/detect_only/verify_only/keyframe_pgo）
  │
Stage 1-2: 关键帧图像保存 → CosPlace 描述子提取（ResNet18+GeM）
  │
Stage 3: 关键帧级检索 → 跨子图 pair 候选
  │
Stage 4: Reloc3R 关键帧对粗位姿估计（保留原始尺度）
  │
Stage 5: RGB-D 深度几何验证（log 尺度搜索 0.1-20×）
  │        三门验收：overlap / RMSE / inlier_ratio
  │
Stage 6: Refine → VerifiedLoopEdge（delta gates vs odometry）
  │
Stage 7: 构建关键帧 Pose Graph
  │        nodes = 所有关键帧 C2W
  │        edges = temporal（相邻 KF）+ handoff（跨子图边界）+ loop（验证通过的闭环）
  │
Stage 8: Open3D LM PGO trial + safety 评估
  │        safety gates: max correction t/r, odom residual ratio, loop residual ratio
  │        鲁棒边剔除（最多 2 次重试）
  │
Stage 9: 如果 accepted → 按关键帧修正轨迹 + 子图 median 修正 Gaussian
  │        保存 keyframe_pgo_result.json
```

---

## 五、坐标约定

| 变量 | 语义 | 写入方 |
|---|---|---|
| `viewpoint.T` | 全局 W2C (4×4) | FrontEnd tracking |
| `inv(viewpoint.T)` | 全局 C2W (4×4) | 按需计算 |
| `seed_global_c2w` | 子图 seed 帧全局 C2W | 切图时设定 |
| `relative_pose` | prev_seed → curr_seed (4×4) | 切图时计算 |
| `correct_tsfm` | PGO/回环修正 (4×4)，左乘 | 仅回环写入 |
| `cam_rot_delta` / `cam_trans_delta` | SE(3) delta for 渲染精化 | 逐帧 Adam |

---

## 六、配置体系

三层配置：三个数据集各有 `base_config.yaml`（通用默认值）+ 场景 `xxx.yaml`（仅覆盖已有参数）。

```text
configs/rgbd/
├── tum/base_config.yaml       (+ fr1_desk.yaml, fr3_office.yaml ...)
├── replica/base_config.yaml   (+ office0.yaml, office0_sp.yaml ...)
└── scannetpp/base_config.yaml (+ 8b5caf3398.yaml ...)
```

### 关键配置组

| 配置组 | 控制范围 |
|---|---|
| `Training.*` | 关键帧、窗口、学习率、densify/prune |
| `FFTEdgeVO.*` | Edge VO 金字塔层级、优化迭代、质量阈值 |
| `Backend.*` | 后端 pose policy、sanity check |
| `Submap.*` | 切图阈值（TUM: 2.0m/80°）、seed init、Handoff |
| `RAP2DGSLite.*` | KNN k、特征权重、选择预算 |
| `LoopClosure.*` | 模式控制、深度验证、PGO safety gates、Reloc3R |
| `opt_params.*` | Gaussian 优化器参数 |
| `Ablation.*` | 功能开关（子图/回环/FDN/FFT/error_mask） |

---

## 七、系统整体数据流图

```text
                            ┌─────────────┐
                            │  RGBD 数据集  │
                            └──────┬──────┘
                                   │
                                   ▼
                      ┌─────────────────────┐
                      │   FrontEnd (主进程)   │
                      │                     │
                      │  频域高通 mask 生成   │
                      │  频域感知 Edge VO    │
                      │  渲染位姿精化         │
                      │  关键帧 + 滑动窗口    │
                      │  子图运动监控/切图    │
                      └──┬──────────────┬───┘
                         │              │
              backend_queue          frontend_queue
              (→ 后端)              (← 后端)
                         │              │
                         ▼              ▲
               ┌──────────────────────────┐
               │   BackEnd (独立进程)       │
               │                          │
               │  Seed 帧地图初始化         │
               │  关键帧扩展 Gaussian      │
               │  RGB+depth+normal 优化    │
               │  Densify / prune         │
               │  子图内 RSKM 随机重放     │
               │  共视 Handoff 选择        │
               │  保存子图 ckpt + 通知     │
               └──┬───────────────────────┘
                  │
                  │ loop_queue ("submap_saved")
                  ▼
      ┌────────────────────────────────────┐
      │  LoopClosureProcess (独立进程)      │
      │  （共视引导子图建图 — 全局一致性）   │
      │                                    │
      │  CosPlace 视觉位置识别              │
      │  关键帧级跨子图检索                  │
      │  Reloc3R 关键帧对粗位姿             │
      │  RGB-D 深度几何验证 + 尺度搜索      │
      │  构建关键帧 Pose Graph             │
      │  PGO trial + safety 评估           │
      │  保存 keyframe_pgo_result.json      │
      └────────────────────────────────────┘
                  │
                  ▼ (程序结束后)
      ┌────────────────────────────────────┐
      │  主进程融合 (slam.py)               │
      │                                    │
      │  加载子图 ckpt                      │
      │  应用 PGO 修正 (rigid_transform)    │
      │  拼接全局 Gaussian                  │
      │  评估 ATE + 渲染质量                │
      └────────────────────────────────────┘
```
