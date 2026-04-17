import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
import json
from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass


class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/image/*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist()
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08): # 图像时间戳(613,)、深度图时间戳(595,)、gt位姿时间戳(2335,)、max_dt表示时间戳允许的最大差值
        associations = []
        for i, t in enumerate(tstamp_image): # i表示图像索引，t表示图像时间戳，i和t同步移动，遍历每个图像时间戳
            if tstamp_pose is None: # 如果没有位姿时间戳，只关联图像和深度图
                j = np.argmin(np.abs(tstamp_depth - t)) #找到与当前图像时间戳t最接近的深度图索引j，并且两者时间差小于 max_dt 才视为关联
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t)) # 所有深度图时间戳减去当前图像时间戳t，取绝对值后找到最小值的索引j，即找到与当前图像时间戳t最接近的深度图索引j(int64类型)
                k = np.argmin(np.abs(tstamp_pose - t)) # 所有位姿时间戳减去当前图像时间戳t，取绝对值后找到最小值的索引k，即找到与当前图像时间戳t最接近的位姿索引k(int64类型)

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations # list of tuples [(16,0,351),(),……,()]，每个元组包含图像索引、深度图索引、位姿索引

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")): # 检测是否存在groundtruth.txt文件
            pose_list = os.path.join(datapath, "groundtruth.txt") # 文件绝对路径，文件内容包含时间戳、平移、四元数
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt") # 文件绝对路径，文件内容包含时间戳及其相对应图像相对路径
        depth_list = os.path.join(datapath, "depth.txt")

        # image、depth、gtpose的时间戳可能不完全对应，需要进行关联
        image_data = self.parse_list(image_list) #(613,2) 第一列时间戳，第二列时间戳对应的图像相对路径
        depth_data = self.parse_list(depth_list) #(595,2) 第一列时间戳，第二列时间戳对应的深度图相对路径
        pose_data = self.parse_list(pose_list, skiprows=1) #字符串类型numpy数组(2335,8) 第一列时间戳，第二到第四列平移，第五到第八列四元数
        pose_vecs = pose_data[:, 0:].astype(np.float64) #转换为浮点型numpy数组

        tstamp_image = image_data[:, 0].astype(np.float64) # 第0列图片时间戳(613,)
        tstamp_depth = depth_data[:, 0].astype(np.float64) # 第0列深度图时间戳(595,)
        tstamp_pose = pose_data[:, 0].astype(np.float64) # 第0列pose时间戳(2335,)


        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose) # list of tuples [(16,0,351),(),……,()]，每个元组包含图像索引、深度图索引、位姿索引
        # 对已关联的帧做下采样（thinning / subsampling）：从原始的连续帧序列中只保留部分帧，使相邻保留帧的时间间隔不小于阈值 1/frame_rate 秒
        # 目的：去除时间上过于密集的冗余帧，降低计算和存储开销，保证均匀的时间间隔。
        indicies = [0] #保存的是被保留的关联项在 associations 列表中的索引（整数列表）
        for i in range(1, len(associations)): # 遍历所有关联的帧
            t0 = tstamp_image[associations[indicies[-1]][0]] # 获取上一个被选择的关联项的图像时间戳
            t1 = tstamp_image[associations[i][0]] # 获取当前关联项的图像时间戳
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies: # 遍历被保留的关联项索引列表
            (i, j, k) = associations[ix] # 获取图像索引i、深度图索引j、位姿索引k
            self.color_paths += [os.path.join(datapath, image_data[i, 1])] # 行索引i 时间戳(列索引0)、相对路径(列索引1)，最终保存的是图像文件的绝对路径
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:] # 行索引k 时间戳(列索引0)、平移(列索引1-3)、四元数(列索引4-7)，(4,)
            trans = pose_vecs[k][1:4] # (3,)
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1)) # 四元数转为(4,4)变换矩阵填充3*3旋转矩阵部分，注意np.roll(quat,1)将四元数循环右移一位从(x,y,z,w)变为(w,x,y,z)
            T[:3, 3] = trans # 将平移部分赋值给变换矩阵的前三行第四列
            self.poses += [np.linalg.inv(T)] #相机位姿c2w

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])), # 挑选出来的rgb的绝对路径
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])), # 挑选出来的depth的绝对路径
                "transform_matrix": (np.linalg.inv(T)).tolist()# 对应的相机位姿矩阵c2w
            }

            self.frames.append(frame)


class ScannetppParser:
    def __init__(self, input_folder, frame_rate=-1):
        self.input_folder = input_folder
        self.color_paths = []
        self.depth_paths = []
        self.poses = []
        self.intrinsics = []

        self.load_poses(input_folder, frame_rate=frame_rate)
        self.n_img = len(self.color_paths)

    def load_poses(self, path, frame_rate=-1):
        pose_intrinsic_imu_json_path = os.path.join(path, "pose_intrinsic_imu.json")
        with open(pose_intrinsic_imu_json_path, "r") as f:
            # iPhone 的 json 结构外层可能不是直接以 frame_name 为 key，具体需参考原始 json
            # 假设其格式类似于 { "frame_00000": {"aligned_poses": [...], "intrinsic": [...], "timestamp": ...} }
            data = json.load(f)

        # RGB 图像位于 rgb 文件夹下
        all_color_paths = sorted(glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg")))
        interval_threshold = 1.0 / frame_rate if frame_rate > 0 else 0.0
        last_timestamp = -float('inf') #更新为当前被保留帧的时间戳，初始值为负无穷，确保第一个帧一定被保留

        for color_path in all_color_paths:
            filename = os.path.basename(color_path)
            frame_name, _ = os.path.splitext(filename)

            if frame_name not in data:
                continue

            frame_data = data[frame_name]

            # 使用 aligned_poses 确保与 Ground Truth Mesh 对齐
            if "aligned_pose" not in frame_data:
                continue

            timestamp = frame_data.get("timestamp", 0.0)

            if timestamp - last_timestamp >= interval_threshold:
                # 读取 c2w 位姿
                T_c2w = np.array(frame_data["aligned_pose"], dtype=np.float32).reshape(4, 4)

                # 你的系统默认读取 w2c，所以要求逆
                T_w2c = np.linalg.inv(T_c2w)

                # 读取 3x3 RGB 内参
                K = np.array(frame_data["intrinsic"], dtype=np.float32).reshape(3, 3)

                # 深度图路径
                depth_path = os.path.join(self.input_folder, "depth", f"{frame_name}.png")

                # 确保深度图确实存在
                if not os.path.exists(depth_path):
                    continue

                self.color_paths.append(color_path)
                self.depth_paths.append(depth_path)
                self.poses.append(T_w2c)
                self.intrinsics.append(K)

                last_timestamp = timestamp


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]
            
            
            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)

            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class ScannetPPDataset(MonocularDataset):
    def __init__(self, args, path, config):
        # 初始化调用基类
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        frame_rate = config["Dataset"].get("frame_rate", 32)

        parser = ScannetppParser(dataset_path, frame_rate=frame_rate)

        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
        self.intrinsics = parser.intrinsics

        # ScanNet++ iPhone Depth 是 16位 PNG, 单位是毫米 (mm)
        self.depth_scale = 1000.0

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        depth_path = self.depth_paths[idx]
        pose = self.poses[idx]

        # 1. 读取离线降采样好的 RGB 图像 (256x192)
        image = np.array(Image.open(color_path))

        # 2. 读取深度图 (256x192)
        # 极度重要：读取 16-bit 图像必须加 cv2.IMREAD_UNCHANGED
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth_raw.astype(np.float32) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)  # (C, H, W)
            .to(device=self.device, dtype=self.dtype)
        )

        # 3. 动态内参自适应缩放 (核心逻辑)
        # 原始 JSON 中的内参是基于 1920x1440 的，我们需要按当前图像尺寸将其缩放下来
        original_w, original_h = 1920, 1440
        current_h, current_w = depth_raw.shape  # 192, 256

        scale_x = current_w / original_w
        scale_y = current_h / original_h

        K_original = self.intrinsics[idx].copy()
        K_new = K_original.copy()

        # 缩放焦距和光心
        K_new[0, 0] *= scale_x  # fx
        K_new[1, 1] *= scale_y  # fy
        K_new[0, 2] *= scale_x  # cx
        K_new[1, 2] *= scale_y  # cy

        fx, fy = K_new[0, 0], K_new[1, 1]
        cx, cy = K_new[0, 2], K_new[1, 2]

        pose = torch.from_numpy(pose).to(device=self.device)

        # 4. 组装返回的内参字典
        intrinsic_dict = {
            "K": K_new,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        }

        return image, depth, pose, intrinsic_dict

class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        
        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True 
        else: 
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
        
        


    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, pose


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    elif config["Dataset"]["type"] == "scannetpp":
        return ScannetPPDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
