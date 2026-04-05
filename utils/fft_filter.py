# utils/fft_filter.py
import cv2
import torch
import numpy as np


class FFTFrequencyFilter:
    def __init__(self, H, W, levels=10):
        self.H = H
        self.W = W
        self.levels = levels
        # CLAHE 用于增强局部对比度，防止全局光照影响高频提取
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.distance, self.step_list = self.prepare_filter_data(H, W)

    def prepare_filter_data(self, height, width):
        pad_W = width // 2
        pad_H = height // 2
        freq_W = width + 2 * pad_W
        freq_H = height + 2 * pad_H
        centerX = freq_W / 2
        centerY = freq_H / 2

        X, Y = np.meshgrid(np.arange(freq_W), np.arange(freq_H))
        distance = np.sqrt((X - centerX) ** 2 + (Y - centerY) ** 2)

        max_dis = np.max(distance)
        step = max_dis / self.levels
        step_list = np.array([step] * self.levels, dtype=np.uint16)

        return torch.tensor(distance, dtype=torch.float32).cuda(), step_list

    def gaussian_hp(self, Dis, D0):
        return 1 - torch.exp(-(Dis ** 2) / (2 * (D0 ** 2)))

    def generate_frequency_mask(self, current_image_bgr):
        """
        输入: BGR 图像 (HxWx3)
        输出: opacity_mask (torch.bool, HxW), True 表示高频区，False 表示低频区
        """
        padded_img = cv2.copyMakeBorder(
            current_image_bgr,
            self.H // 2, self.H // 2, self.W // 2, self.W // 2,
            cv2.BORDER_REFLECT_101
        )

        gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        gray_tensor = torch.tensor(gray, dtype=torch.float32).cuda()
        freq = torch.fft.fft2(gray_tensor, dim=(0, 1), norm='ortho')
        freq = torch.fft.fftshift(freq, dim=(0, 1))

        # 提取第一层高通滤波器 (FGS-SLAM 实际单尺度逻辑)
        D0 = float(self.step_list[0])
        filter_g = self.gaussian_hp(self.distance, D0=D0)

        freq_g = freq * filter_g
        freq_g = torch.fft.ifftshift(freq_g, dim=(0, 1))

        gray_g = torch.fft.ifft2(freq_g, dim=(0, 1), norm='ortho')
        gray_g = torch.abs(gray_g)

        # 裁剪回原图尺寸
        gray_g = gray_g[self.H // 2: self.H // 2 + self.H,
                 self.W // 2: self.W // 2 + self.W]

        # OpenCV Triangle 阈值二值化
        gray_g_np = gray_g.cpu().numpy()
        gray_g_np = cv2.normalize(gray_g_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_np = cv2.threshold(gray_g_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        opacity_mask = torch.tensor(mask_np > 0, dtype=torch.bool).cuda()
        return opacity_mask