import torch
from torch import nn
import numpy as np
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
from utils.slam_utils import image_gradient, image_gradient_mask
import torch.nn.functional as F
from utils.normal_utils import intrins_to_intrins_inv, get_cam_coords, d2n_tblr

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        dynamic_intrinsic,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0"
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        # T = torch.eye(4, device=device)
        # self.R = T[:3, :3]
        # self.T = T[:3, 3]
        # self.R_gt = gt_T[:3, :3]
        # self.T_gt = gt_T[:3, 3]
        self.T = torch.eye(4, device=device).to(torch.float32) #被初始化为单位矩阵，表示当前（估计的）相机位姿
        self.T_gt = gt_T.to(device=device).to(torch.float32).clone() #4*4 matrix


        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        # 是否将该帧作为子图局部坐标系原点并固定其位姿。
        # 对于这类帧，只允许优化高斯与曝光，不允许通过相机增量更新位姿。
        self.fixed_pose = False

        # --- B1/B2: submap seed soft-anchor state ---
        self.is_submap_seed = False

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # ---- PAR RSKM metadata (plain attrs, survive pickle) ----
        self.vo_init_c2w = None             # np.ndarray (4,4) float64, C2W from VO prior
        self.render_opt_c2w = None          # np.ndarray (4,4) float64, C2W after tracking
        self.par_pose_trans_error = None    # float, VO vs render translation error (m)
        self.par_pose_rot_error_deg = None  # float, VO vs render rotation error (deg)
        self.par_tracking_loss = None       # float, best_loss from tracking refinement
        self.par_reliability = None         # float, r_i = exp(-beta * pose_error)
        self.par_replay_count = 0           # int, times selected for replay
        self.par_last_replay_iter = -1      # int, last mapping iteration when replayed
        self.par_initialized = False        # bool, whether PAR metadata has been computed
        # 兼容处理：如果未提供动态内参，则根据传入的静态参数构造一个 3x3 矩阵
        # 防止下游 SLAM 跟踪线程调用 self.dynamic_intrinsic 时引发 NoneType 错误
        if dynamic_intrinsic is not None:
            self.dynamic_intrinsic = dynamic_intrinsic.clone().detach().to(device=device, dtype=torch.float32)
        else:
            self.dynamic_intrinsic = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
                device=device,
                dtype=torch.float32
            )
        self.projection_matrix = projection_matrix.to(device=device)

        intrins = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.intrins_inv = intrins_to_intrins_inv(intrins).float().unsqueeze(0).to(device)


    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        # 动态解析从 Dataset 中获取的元组, data tuple(TUMdataset) (color, depth, pose 从groundtruth文件中读取的相机位姿求逆后传入)
        data = dataset[idx]
        if len(data) == 3:
            gt_color, gt_depth, gt_pose = data
            intrinsic_dict = None
        elif len(data) == 4:
            gt_color, gt_depth, gt_pose, intrinsic_dict = data
        else:
            raise ValueError(f"Dataset returned {len(data)} items, expected 3 or 4.")

        # 默认使用数据集对象上的静态全局内参
        fx = dataset.fx
        fy = dataset.fy
        cx = dataset.cx
        cy = dataset.cy
        fovx = dataset.fovx
        fovy = dataset.fovy
        proj_mat = projection_matrix
        dynamic_intrinsic_tensor = None

        # 如果存在逐帧动态内参，则覆盖静态参数，并重新计算投影矩阵
        if intrinsic_dict is not None:
            fx = intrinsic_dict["fx"]
            fy = intrinsic_dict["fy"]
            cx = intrinsic_dict["cx"]
            cy = intrinsic_dict["cy"]

            # 使用新的焦距重新计算视场角 (Field of View)
            fovx = focal2fov(fx, dataset.width)
            fovy = focal2fov(fy, dataset.height)

            # 根据当前帧的新内参，动态生成用于高斯渲染的投影矩阵
            proj_mat = getProjectionMatrix2(
                znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=dataset.width, H=dataset.height
            ).transpose(0, 1).to(dataset.device)

            # 解析 3x3 K 矩阵
            K = intrinsic_dict["K"]
            if isinstance(K, np.ndarray):
                dynamic_intrinsic_tensor = torch.from_numpy(K)
            else:
                dynamic_intrinsic_tensor = torch.tensor(K)

        return Camera(
            uid=idx,
            color=gt_color,
            depth=gt_depth,
            gt_T=gt_pose,
            dynamic_intrinsic=dynamic_intrinsic_tensor,
            projection_matrix=proj_mat,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            fovx=fovx,
            fovy=fovy,
            image_height=dataset.height,
            image_width=dataset.width,
            device=dataset.device
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)

        # 使用明确的关键字传参，避免位置错乱
        return Camera(
            uid=uid,
            color=None,
            depth=None,
            gt_T=T,
            dynamic_intrinsic=None,  # <--- 必须在这里补上 None
            projection_matrix=projection_matrix,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            fovx=FoVx,
            fovy=FoVy,
            image_height=H,
            image_width=W
        )

    def reset_pose_deltas(self):
        if self.cam_rot_delta is not None:
            self.cam_rot_delta.data.zero_()
        if self.cam_trans_delta is not None:
            self.cam_trans_delta.data.zero_()

    @property
    def world_view_transform(self):
        # return getWorld2View2(self.R, self.T).transpose(0, 1)
        return self.T.transpose(0, 1).to(device=self.device)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    # 修改为：
    @property
    def camera_center(self):
        # 正确计算相机在世界坐标系下的 3D 中心点
        return self.world_view_transform.inverse()[3, :3]

    #grad_mask 管"在哪里优化位姿"，freq_mask 管"新高斯点怎么撒、撒多大"，error_mask 管"在哪里补新高斯点"
    # 该梯度掩码在计算仅基于 RGB 颜色 的相机跟踪（Tracking）损失时使用
    # #该函数生成了一个二值掩码 self.grad_mask，掩码中的 True (或 1) 表示该像素点位于边缘或纹理区域，
    # 将被用于后续的 Tracking 损失计算，忽略平坦区域以提高计算效率和稳定性
    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]
        #主要作用是计算图像的梯度掩码（Gradient Mask），用于筛选出图像中纹理丰富或边缘明显的区域。  这些区域在视觉 SLAM 或相机跟踪（Tracking）算法中非常重要，因为还可以基于这些具体的特征点计算光度误差，而平坦无纹理的区域通常无法提供有效的几何约束。
        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2) #主要目的是计算当前相机图像的梯度强度图（Gradient Intensity Map）。这通常用于识别图像中的边缘或纹理丰富区域，这些区域在 SLAM 跟踪或计算光度误差时往往更重要。

        if config["Dataset"]["type"] == "replica":
            # row, col = 32, 32 # 将图像划分为32x32个小块
            size = 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            I = img_grad_intensity.unsqueeze(0)
            I_unf = F.unfold(I, size, stride=size)
            median_patch, _ = torch.median(I_unf, dim=1, keepdim=True)
            mask = (I_unf > (median_patch * multiplier)).float()
            I_f = F.fold(mask, I.shape[-2:], size, stride=size).squeeze(0)
            self.grad_mask = I_f
            # for r in range(row): #遍历每个图像块
            #     for c in range(col):
            #         block = img_grad_intensity[
            #             :,
            #             r * int(h / row) : (r + 1) * int(h / row),
            #             c * int(w / col) : (c + 1) * int(w / col),
            #         ]
            #         th_median = block.median() #计算该块内梯度强度的中位数
            #         block[block > (th_median * multiplier)] = 1 #大于该阈值的像素置为 1，否则置为 0
            #         block[block <= (th_median * multiplier)] = 0
            # self.grad_mask = img_grad_intensity #这种局部自适应方法能更好地处理光照不均匀或纹理分布不均的情况，确保在图像的各个区域都能提取到相对显著的特征点
        else: #全局阈值法，适用于纹理分布较均匀的图像
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            ) #只有梯度强度显著高于全局中位数的像素点（即边缘或纹理丰富点）会被保留（标记为 True），平坦区域会被过滤掉。该掩码随后用于相机追踪时的损失计算

        gt_image = self.original_image.cuda()
        _, h, w = self.original_image.cuda().shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = 0.05
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(
            *mask_shape
        )
        self.rgb_pixel_mask = rgb_pixel_mask * self.grad_mask
        self.rgb_pixel_mask_mapping = rgb_pixel_mask

        if self.depth is not None:
            self.gt_depth = torch.from_numpy(self.depth).to(
                dtype=torch.float32, device=self.device
            )[None]

            depth = self.gt_depth.unsqueeze(0)
            points = get_cam_coords(self.intrins_inv, depth) #右x、下y、深度z OpenCV相机坐标系
            normal, valid_mask = d2n_tblr(points, d_min=1e-3, d_max=1000.0)
            normal = normal * valid_mask
            self.normal = normal #(1,3,H,W)相机坐标系下的法向量
            self.normal_raw = self.normal.squeeze(0).permute(1, 2, 0).cpu().numpy() #(H,W,3)相机坐标系下的法向量
            self.mask = valid_mask

        if self.mask is not None:
            self.rgb_pixel_mask = self.rgb_pixel_mask * self.mask
            self.rgb_pixel_mask_mapping = self.rgb_pixel_mask_mapping


    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None
        #子图第一帧
        self.fixed_pose = False
        self.is_submap_seed = False

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

        self.rgb_pixel_mask = None
        self.rgb_pixel_mask_mapping = None
        self.gt_depth = None
        self.normal = None
        self.normal_raw = None

    # utils/camera_utils.py
    def release_mapping_payload(self):
        """
        只释放建图/监督相关的大张量，
        保留位姿、内参、曝光参数、pose delta，不影响后续 render / refinement。
        """
        self.original_image = None
        self.depth = None
        self.grad_mask = None
        self.rgb_pixel_mask = None
        self.rgb_pixel_mask_mapping = None
        self.gt_depth = None
        self.normal = None
        self.normal_raw = None
        self.mask = None

        if hasattr(self, "error_mask"):
            self.error_mask = None

class CameraMsg:
    def __init__(self, Camera=None, uid=None, T=None, T_gt=None):
        if Camera is not None:
            self.uid = Camera.uid
            self.T = Camera.T
            self.T_gt = Camera.T_gt
        else:
            self.uid = uid
            self.T = T
            self.T_gt = T_gt
