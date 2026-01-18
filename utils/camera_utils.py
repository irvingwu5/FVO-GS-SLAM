import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
        label_info=None,
        plane_param_info=None
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

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
        self.label_info = label_info # dict:3{num_planes int,label_data(ndarray(H,W)),nonplanepxl_mask(ndarray(H,W))}
        self.plane_param_info = plane_param_info

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
    #Tensor(3,H,W)、ndarray(H,W)、Tensor(4,4)、dict:3{num_planes int,label_data(ndarray(H,W)),nonplanepxl_mask(ndarray(H,W))},plane_param_info dict:3{plane_normals ndarray(num_planes,3),plane_center ndarray(num_planes,3), plane_equations ndarray(num_planes,4)}
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose, label_info, plane_param_info = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
            label_info=label_info,
            plane_param_info=plane_param_info
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device) #接收新的旋转矩阵 R 和平移向量 t，并将它们更新到当前相机对象的属性中。
        self.T = t.to(device=self.device)
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
            row, col = 32, 32 # 将图像划分为32x32个小块
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row): #遍历每个图像块
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median() #计算该块内梯度强度的中位数
                    block[block > (th_median * multiplier)] = 1 #大于该阈值的像素置为 1，否则置为 0
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity #这种局部自适应方法能更好地处理光照不均匀或纹理分布不均的情况，确保在图像的各个区域都能提取到相对显著的特征点
        else: #全局阈值法，适用于纹理分布较均匀的图像
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            ) #只有梯度强度显著高于全局中位数的像素点（即边缘或纹理丰富点）会被保留（标记为 True），平坦区域会被过滤掉。该掩码随后用于相机追踪时的损失计算

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
