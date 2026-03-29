import pathlib
import threading
import time
from datetime import datetime

import cv2
import glfw
import imgviz
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from OpenGL import GL as gl
from matplotlib.colors import hsv_to_rgb
import copy
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import fov2focal
from gui.gl_render import util, util_gau
from gui.gl_render.render_ogl import OpenGLRenderer
from gui.gui_utils import (
    GaussianPacket,
    Packet_vis2main,
    create_frustum,
    cv_gl,
    get_latest_queue,
    depth_to_normal,
    vfov_to_hfov,
)
from utils.camera_utils import Camera
from utils.logging_utils import Log

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import natsort
from gaussian_splatting.utils.sh_utils import RGB2SH


class SLAM_GUI:
    def __init__(self, params_gui=None):
        self.step = 0
        self.process_finished = False
        self.device = "cuda"

        self.frustum_dict = {}
        self.model_dict = {}
        self.trj_dict = {}

        self.init_widget()

        self.q_main2vis = None
        self.gaussian_cur = None
        self.gaussian_motion = None
        self.pipe = None
        self.background = None

        self.init = False
        self.kf_window = None
        self.render_img = None

        self.is_2dgs = False
        self.render_fps = 50.0
        self.ts_prev = 0.0
        self.fid_dict = {}
        self.fid_cur = 0
        self.frustum_dict_replay = {}

        if params_gui is not None:
            self.background = params_gui.background
            self.gaussian_cur = params_gui.gaussians
            self.init = True
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.pipe = params_gui.pipe
            self.is_2dgs = params_gui.is_2dgs

            self.warpfield = None
            self.warpfield_warmup = False
            self.seq_name = params_gui.seq_name

        self.datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.gaussian_nums = []

        self.timestamp_end = 0.1
        self.t0 = time.time()
        self.ts_prev = 0.0
        self.replay_length_sec = 4

        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw()
        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)

        self.save_path = "."
        self.save_path = pathlib.Path(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.origin_pose = None
        threading.Thread(target=self._update_thread).start()

    def init_widget(self):
        self.window_w, self.window_h = 1600, 900

        self.window = gui.Application.instance.create_window(
            "2DGS SLAM", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)

        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"
        self.lit.line_width = 2.0
        self.lit.transmission = 1.0

        self.lit_replay = rendering.MaterialRecord()
        self.lit_replay.shader = "unlitLine"
        self.lit_replay.line_width = 6.0
        self.lit_replay.transmission = 1.0

        self.lit_trj = rendering.MaterialRecord()
        self.lit_trj.shader = "unlitLine"
        self.lit_trj.line_width = 6.0
        self.lit_trj.transmission = 0.0

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"

        self.specular_geo = rendering.MaterialRecord()
        self.specular_geo.shader = "defaultLit"

        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        em = self.window.theme.font_size
        margin = 0.1 * em
        self.panel = gui.Vert(0.1 * em, gui.Margins(margin))
        self.button = gui.ToggleSwitch("Resume/Pause")
        self.button.is_on = True
        self.button.set_on_clicked(self._on_button)
        self.panel.add_child(self.button)

        ## Camera options
        collapse_cam = gui.CollapsableVert(
            "Camera Options", 0.1 * em, gui.Margins(em, 0, 0, 0)
        )
        self.panel.add_child(collapse_cam)

        viewpoint_tile = gui.Horiz(0.1 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.1 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.1 * em, gui.Margins(margin))

        ##Check boxes
        vp_subtile1.add_child(gui.Label("Follow options"))
        chbox_tile = gui.Horiz(0.1 * em, gui.Margins(margin))
        self.followcam_chbox = gui.Checkbox("Follow")
        self.followcam_chbox.checked = True
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)
        vp_subtile1.add_child(chbox_tile)

        ##Combo panels
        combo_tile = gui.Horiz(0.1 * em, gui.Margins(margin))

        ## Jump to the camera viewpoint
        self.combo_kf = gui.Combobox()
        self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        combo_tile.add_child(gui.Label("Viewpoint list"))
        combo_tile.add_child(self.combo_kf)

        combo_subtile3 = gui.Vert(0.1 * em, gui.Margins(margin))
        self.combo_waypoints = gui.Combobox()
        self.waypoints_dict = dict()
        self.waypoints_dict_save = dict()
        combo_subtile3.add_child(gui.Label("Waypoints"))
        combo_subtile3.add_child(self.combo_waypoints)
        combo_tile.add_child(combo_subtile3)

        vp_subtile2.add_child(combo_tile)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        collapse_cam.add_child(viewpoint_tile)

        chbox_tile_3dobj = gui.Horiz(0.1 * em, gui.Margins(margin))
        collapse_cam.add_child(chbox_tile_3dobj)

        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.kf_window_chbox = gui.Checkbox("Active window")
        self.kf_window_chbox.set_on_checked(self._on_kf_window_chbox)
        chbox_tile_3dobj.add_child(self.kf_window_chbox)

        self.axis_chbox = gui.Checkbox("Axis")
        self.axis_chbox.checked = False
        self.axis_chbox.set_on_checked(self._on_axis_chbox)
        chbox_tile_3dobj.add_child(self.axis_chbox)

        collapse_rendering = gui.CollapsableVert(
            "Rendering Options", 0.1 * em, gui.Margins(em, 0, 0, 0)
        )

        self.panel.add_child(collapse_rendering)

        self.chbox_tile_geometry = gui.Horiz(0.1 * em, gui.Margins(margin))
        collapse_rendering.add_child(self.chbox_tile_geometry)

        self.depth_chbox = gui.Checkbox("Depth")
        self.depth_chbox.checked = False
        self.chbox_tile_geometry.add_child(self.depth_chbox)

        self.normals_chbox = gui.Checkbox("Normal")
        self.normals_chbox.checked = False
        self.chbox_tile_geometry.add_child(self.normals_chbox)

        # ====== 加上下面这三行，恢复椭球体渲染选项 ======
        self.elipsoid_chbox = gui.Checkbox("Elipsoid")
        self.elipsoid_chbox.checked = False
        self.chbox_tile_geometry.add_child(self.elipsoid_chbox)
        # ================================================

        slider_tile_gaussian = gui.Horiz(0.1 * em, gui.Margins(margin))
        slider_label = gui.Label("Gaussian Size Scale")
        self.scaling_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scaling_slider.set_limits(0.001, 1.0)
        self.scaling_slider.double_value = 1.0
        slider_tile_gaussian.add_child(slider_label)
        slider_tile_gaussian.add_child(self.scaling_slider)
        collapse_rendering.add_child(slider_tile_gaussian)

        collapse_playback = gui.CollapsableVert(
            "Playback Options (for Offline Visualization)",
            0.33 * em,
            gui.Margins(em, 0, 0, 0),
        )

        self.panel.add_child(collapse_playback)

        chbox_tile_replay = gui.Horiz(0.1 * em, gui.Margins(margin))
        self.replay_btn = gui.Button("          Replay          ")
        self.replay_btn.background_color = gui.Color(0.0, 0.5, 0.1)
        self.replay_btn.set_on_clicked(self._on_replay_btn)
        self.stop_btn = gui.Button("          Stop          ")
        self.stop_btn.background_color = gui.Color(0.5, 0.0, 0.1)
        self.stop_btn.set_on_clicked(self._on_stop_btn)
        chbox_tile_replay.add_child(self.replay_btn)
        chbox_tile_replay.add_child(self.stop_btn)
        collapse_playback.add_child(chbox_tile_replay)

        chbox_tile_trj = gui.Horiz(0.1 * em, gui.Margins(margin))


        self.trj_replay_chbox = gui.Checkbox("Replay Camera")
        self.trj_replay_chbox.checked = False
        self.trj_replay_chbox.set_on_checked(self._on_trj_replay_chbox)
        chbox_tile_trj.add_child(self.trj_replay_chbox)

        self.trj_chbox = gui.Checkbox("Trj")
        self.trj_chbox.checked = False
        self.trj_chbox.set_on_checked(self._on_trj_chbox)
        chbox_tile_trj.add_child(self.trj_chbox)

        self.replaycam_chbox = gui.Checkbox("Frustum")
        self.replaycam_chbox.checked = False
        self.replaycam_chbox.set_on_checked(self._on_replaycam_chbox)
        chbox_tile_trj.add_child(self.replaycam_chbox)
        collapse_playback.add_child(chbox_tile_trj)

        play_back_tile = gui.Horiz(0.1 * em, gui.Margins(margin))
        slider_tile = gui.Horiz(0.1 * em, gui.Margins(margin))
        slider_label = gui.Label("Current Time Scale")
        self.timescale_slider = gui.Slider(gui.Slider.DOUBLE)
        self.timescale_slider.set_limits(0, 1.0)
        self.timescale_slider.double_value = 1.0
        slider_tile.add_child(slider_label)
        slider_tile.add_child(self.timescale_slider)
        play_back_tile.add_child(slider_tile)

        collapse_playback.add_child(play_back_tile)

        save_load_tile = gui.Horiz(0.1 * em, gui.Margins(margin))
        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(self._on_screenshot_btn)
        save_load_tile.add_child(self.screenshot_btn)

        self.screenrecord_btn = gui.Button("ScreenRecord")
        self.is_recording = False
        self.record_video = []
        self.screenrecord_btn.set_on_clicked(self._on_screenrecord_btn)
        save_load_tile.add_child(self.screenrecord_btn)

        self.panel.add_child(save_load_tile)

        ## Rendering Tab
        tab_margins = gui.Margins(0, int(np.round(0.1 * em)), 0, 0)
        tabs = gui.TabControl()

        tab_info = gui.Vert(0, tab_margins)
        self.output_info = gui.Label("Number of Gaussians: ")

        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()

        tab_info.add_child(self.in_rgb_widget)
        tab_info.add_child(self.in_depth_widget)

        dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)
        dummy_img_o3d = o3d.geometry.Image(dummy_img)
        self.in_rgb_widget.update_image(dummy_img_o3d)
        self.in_depth_widget.update_image(dummy_img_o3d)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)
        self.window.add_child(self.panel)

    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window

    def update_activated_renderer_state(self, gaus):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.scaling_slider.double_value)
        self.g_renderer.set_render_mod(-4)
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)

    def add_camera(self, camera, name, color=[0, 1, 0], gt=False, size=0.005):
        W2C = camera.T_gt.clone() if gt else camera.T.clone()
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color)
            self.combo_kf.add_item(name)
            self.frustum_dict[name] = frustum
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        return frustum

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 0.7
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _on_combo_kf(self, new_val, new_idx):
        frustum = (
            self.frustum_dict[new_val]
            if new_val != "replay"
            else self.frustum_dict_replay[new_val]
        )
        viewpoint = (
            frustum.view_dir_behind
            if self.staybehind_chbox.checked
            else frustum.view_dir
        )

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_replaycam_chbox(self, is_checked):
        if "replay" in self.frustum_dict_replay.keys():
            self.widget3d.scene.show_geometry("replay", is_checked)

    def _on_replay_btn(self):
        self.followcam_chbox.checked = False
        self.staybehind_chbox.checked = False
        self.cameras_chbox.checked = False
        self._on_cameras_chbox(False)
        self.trj_chbox.checked = True
        self._on_trj_chbox(True)
        self.replaycam_chbox.checked = True
        self._on_replaycam_chbox(True)
        self.trj_replay_chbox.checked = True

    def _on_stop_btn(self):
        self.followcam_chbox.checked = False
        self.staybehind_chbox.checked = False
        self.cameras_chbox.checked = False
        self._on_cameras_chbox(False)
        self.trj_chbox.checked = True
        self._on_trj_chbox(True)
        self.replaycam_chbox.checked = True
        self._on_replaycam_chbox(True)
        self._on_trj_replay_chbox(True)
        self.trj_replay_chbox.checked = False

    def _on_record_btn(self):
        self.record_btn_on = not self.record_btn_on
        self._on_screenrecord_btn()
        self._on_replay_btn()

    def _on_trj_chbox(self, is_checked):
        if not self.frustum_dict:
            return

        names = natsort.natsorted(self.frustum_dict.keys())

        for idx in range(1, len(names) - 1):
            frame1 = self.frustum_dict[names[idx]].view_dir[1]
            frame2 = self.frustum_dict[names[idx + 1]].view_dir[1]

            points = [frame1, frame2]
            lines = [[0, 1]]
            colors = [[0.05, 0.55, 0.0]]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            name = f"trj_{idx}"

            if is_checked:
                self.widget3d.scene.remove_geometry(name)
                self.widget3d.scene.add_geometry(name, line_set, self.lit_trj)
            else:
                self.widget3d.scene.remove_geometry(name)


    def _on_trj_replay_chbox(self, is_checked):
        names = natsort.natsorted(list(self.fid_dict.keys()))
        if not names:
            return

        def orthogonalize_rotation_matrix(R):
            try:
                U, _, Vt = np.linalg.svd(R)
                R_ortho = U @ Vt
                if np.linalg.det(R_ortho) < 0:
                    U[:, -1] *= -1
                    R_ortho = U @ Vt
            except Exception:
                R_ortho = R
            return R_ortho

        for i in range(len(names) - 1):
            cur_fid = names[i]
            next_fid = names[i + 1]

            if self.fid_cur < cur_fid:
                return
            if cur_fid <= self.fid_cur <= next_fid:
                pose1 = self.fid_dict[cur_fid].view_dir[-1]
                pose2 = self.fid_dict[next_fid].view_dir[-1]

                alpha = (self.fid_cur - cur_fid) / (next_fid - cur_fid)
                pose_intp = pose1 + (pose2 - pose1) * alpha
                pose_intp[0:3, 0:3] = orthogonalize_rotation_matrix(pose_intp[0:3, 0:3])
                break
        else:
            return

        if "replay" not in self.frustum_dict_replay:
            frustum = create_frustum(
                pose_intp, frusutum_color=[0.5, 0.0, 0], size=0.025
            )
            self.frustum_dict_replay["replay"] = frustum
            self.widget3d.scene.add_geometry(
                "replay", frustum.line_set, self.lit_replay
            )

        frustum = self.frustum_dict_replay["replay"]
        frustum.update_pose(pose_intp)
        self.widget3d.scene.set_geometry_transform(
            "replay", pose_intp.astype(np.float64)
        )
        self.widget3d.scene.show_geometry("replay", self.replaycam_chbox.checked)

    def _on_axis_chbox(self, is_checked):
        name = "axis"
        if is_checked:
            self.widget3d.scene.remove_geometry(name)
            self.widget3d.scene.add_geometry(name, self.axis, self.lit_geo)
        else:
            self.widget3d.scene.remove_geometry(name)

    def _on_kf_window_chbox(self, is_checked):
        if self.kf_window is None:
            return
        edge_cnt = 0
        for key in self.kf_window.keys():
            for kf_idx in self.kf_window[key]:
                name = "kf_edge_{}".format(edge_cnt)
                edge_cnt += 1
                if "keyframe_{}".format(key) not in self.frustum_dict.keys():
                    continue
                if "keyframe_{}".format(kf_idx) not in self.frustum_dict.keys():
                    continue
                test1 = self.frustum_dict["keyframe_{}".format(key)].view_dir[1]
                kf = self.frustum_dict["keyframe_{}".format(kf_idx)].view_dir[1]
                points = [test1, kf]
                lines = [[0, 1]]
                colors = [[0, 1, 0]]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                if is_checked:
                    self.widget3d.scene.remove_geometry(name)
                    self.widget3d.scene.add_geometry(name, line_set, self.lit)
                else:
                    self.widget3d.scene.remove_geometry(name)

    def _on_button(self, is_on):
        packet = Packet_vis2main()
        packet.flag_pause = not self.button.is_on
        self.q_vis2main.put(packet)

    def _on_slider(self, value):
        packet = self.prepare_viz2main_packet()
        self.q_vis2main.put(packet)

    def _on_screenshot_btn(self):
        if self.render_img is None:
            return

        save_dir = self.save_path / "screenshots" / self.seq_name / self.datetime
        save_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(save_dir.glob("*-gui*.png"))) + 1
        # create the filename
        filename = save_dir / "screenshot"
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}-gui-{idx}.png", img)
        img = np.asarray(self.render_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}-{idx}.png", img)
        print("Screenshot saved to", f"{filename}-gui-{idx}.png")

    def _on_screenrecord_btn(self, opt_type=""):
        self.is_recording = not self.is_recording

        if not self.is_recording:
            self.video_end_time = time.time()
            save_dir = self.save_path / "screenrecords" / self.seq_name / self.datetime
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / (opt_type + "video.mp4")

            width = int(self.widget3d_width)
            height = int(self.window.size.height)
            frame_size = (width, height)

            fps = len(self.record_video) / (self.video_end_time - self.video_start_time)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(filename), fourcc, fps, frame_size)

            for frame in self.record_video:
                out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            out.release()
            cv2.destroyAllWindows()
            self.record_video = []
            Log("Recorded video saved to {}".format(filename), tag="GUI")
        else:
            self.video_start_time = time.time()

    def screenrecord(self):
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(
            app.render_to_image(
                self.widget3d.scene, self.widget3d_width, self.window.size.height
            )
        )
        self.record_video.append(img)

    def receive_data(self, q):
        if q is None:
            return

        gaussian_packet = get_latest_queue(q)
        if gaussian_packet is None:
            return

        if gaussian_packet.has_gaussians:
            self.gaussian_cur = gaussian_packet
            self.init = True

        if gaussian_packet.current_frame is not None:
            frustum = self.add_camera(
                gaussian_packet.current_frame, name="current", color=[0, 1, 0]
            )
            # 替换为这一行（使用 getattr 保证即使没有 uid 也不会闪退）：
            self.timestamp_end = max(self.timestamp_end, getattr(gaussian_packet.current_frame, 'uid', 0))
            if self.followcam_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        if gaussian_packet.keyframes is not None:
            for keyframe in gaussian_packet.keyframes:
                name = "keyframe_{}".format(keyframe.uid)
                frustum = self.add_camera(keyframe, name=name, color=[0, 0, 1])
                self._on_trj_chbox(is_checked=self.trj_chbox.checked)
                # 替换为下面这两行（直接使用 uid）：
                self.fid_dict[keyframe.uid] = frustum
                self.timestamp_end = max(self.timestamp_end, keyframe.uid)

        if gaussian_packet.kf_window is not None:
            self.kf_window = gaussian_packet.kf_window
            self._on_kf_window_chbox(is_checked=self.kf_window_chbox.checked)

        if gaussian_packet.gtcolor is not None:
            rgb = torch.clamp(gaussian_packet.gtcolor, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if gaussian_packet.gtdepth is not None:
            depth = gaussian_packet.gtdepth
            depth = imgviz.depth2rgb(
                depth, min_value=0.1, max_value=5.0, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(depth)
            self.in_depth_widget.update_image(rgb)


        if gaussian_packet.finish:
            Log("Received terminate signal", tag="GUI")
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True

    def get_current_cam(self):
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()

        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        T = torch.from_numpy(w2c).to("cuda").to(torch.float32)
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_gui.shape[1],
            W=image_gui.shape[2],
        )
        current_cam.T = T.clone()
        return current_cam

    def normalize_tensor(
        self,
        tensor,
        min_val=torch.tensor([-0.1, -0.1, -0.1], device="cuda"),
        max_val=torch.tensor([0.1, 0.1, 0.1], device="cuda"),
    ):
        if min_val is None:
            min_val = tensor.min(dim=0, keepdim=True).values
        if max_val is None:
            max_val = tensor.max(dim=0, keepdim=True).values

        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        normalized_tensor = torch.clamp(normalized_tensor, 0.0, 1.0)
        return normalized_tensor

    def rasterize(self, current_cam):
        ts_now = time.perf_counter()
        self.ts_prev = ts_now
        self.time_now = time.time()


        fid = torch.tensor(
            self.timescale_slider.double_value * self.timestamp_end,
            dtype=torch.float32,
            device="cuda",
        )

        self.fid_cur = fid.cpu().numpy()

        # 判空保护
        if self.gaussian_cur is None or self.gaussian_cur.get_xyz.shape[0] <= 0:
            return None, [None, None, None]

        # 直接调用你提供的 2DGS render 函数
        # 注意参数：匹配 viewpoint_camera, pc, pipe, bg_color, scaling_modifier, surf
        rendering_data = render(
            viewpoint_camera=current_cam,
            pc=self.gaussian_cur,
            pipe=self.pipe,
            bg_color=self.background,
            scaling_modifier=self.scaling_slider.double_value,
            surf=self.normals_chbox.checked
        )

        # 静态场景没有 d_xyz 等形变数据，直接返回 None
        return rendering_data, [None, None, None]

    def render_o3d_image(self, results, current_cam, deformation_data=None):
        if self.depth_chbox.checked:
            depth = results["depth"]
            depth = depth[0, :, :].detach().cpu().numpy()
            max_depth = np.max(depth)
            depth = imgviz.depth2rgb(
                depth, min_value=0.1, max_value=max_depth, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(depth)

        elif self.normals_chbox.checked:
            if self.is_2dgs:
                if self.origin_pose is None:
                    self.origin_pose = current_cam.T[0:3, 0:3].float()
                normal = results["rend_normal"].view(3, -1)
                normal_world = self.origin_pose @ normal.float()
                normal_world = normal_world.view(
                    3, current_cam.image_height, current_cam.image_width
                )
                normal_world = normal_world * 0.5 + 0.5
                normal_rgb = (
                    (torch.clamp(normal_world, min=0, max=1.0) * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                render_img = o3d.geometry.Image(normal_rgb)

            else:
                depth = results["depth"]
                depth = depth

                rgb = (
                    (torch.clamp(results["render"], min=0, max=1.0) * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                rgb = o3d.geometry.Image(rgb)
                depth = o3d.geometry.Image(
                    depth.detach().cpu().numpy().astype(np.float32)[0]
                )
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb, depth, depth_scale=1.0, depth_trunc=10.0
                )
                image_gui = torch.zeros(
                    (1, int(self.window.size.height), int(self.widget3d_width))
                )
                vfov_deg = self.widget3d.scene.camera.get_field_of_view()
                hfov_deg = vfov_to_hfov(
                    vfov_deg, image_gui.shape[1], image_gui.shape[2]
                )
                FoVx = np.deg2rad(hfov_deg)
                FoVy = np.deg2rad(vfov_deg)
                fx = fov2focal(FoVx, current_cam.image_width)
                fy = fov2focal(FoVy, current_cam.image_height)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd,
                    o3d.camera.PinholeCameraIntrinsic(
                        current_cam.image_width,
                        current_cam.image_height,
                        fx,
                        fy,
                        current_cam.image_width // 2,
                        current_cam.image_height // 2,
                    ),
                    project_valid_depth_only=False,
                )

                pcd = (
                    torch.from_numpy(np.asarray(pcd.points))
                    .view(1, current_cam.image_height, current_cam.image_width, 3)
                    .permute(0, 3, 1, 2)
                )
                normals, mask = depth_to_normal(pcd, k=3, d_min=0.1, d_max=10.0)
                normals_rgb = 0.5 * normals[0] * mask[0]
                normals_rgb = (normals_rgb + 0.5) * 255
                normals_rgb = (
                    normals_rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
                )
                render_img = o3d.geometry.Image(normals_rgb)


        elif self.elipsoid_chbox.checked:

            if self.gaussian_cur is None:
                return

            glfw.poll_events()

            gl.glClearColor(0, 0, 0, 1.0)

            gl.glClear(

                gl.GL_COLOR_BUFFER_BIT

                | gl.GL_DEPTH_BUFFER_BIT

                | gl.GL_STENCIL_BUFFER_BIT

            )

            w = int(self.window.size.width * self.widget3d_width_ratio)

            glfw.set_window_size(self.window_gl, w, self.window.size.height)

            self.g_camera.fovy = current_cam.FoVy

            self.g_camera.update_resolution(self.window.size.height, w)

            self.g_renderer.set_render_reso(w, self.window.size.height)

            # 恢复原始的视锥提取代码（保证鼠标拖拽 Open3D 窗口时，底层椭球也能跟着转）

            frustum = create_frustum(

                np.linalg.inv(cv_gl @ self.widget3d.scene.camera.get_view_matrix())

            )

            self.g_camera.position = frustum.eye.astype(np.float32)

            self.g_camera.target = frustum.center.astype(np.float32)

            self.g_camera.up = frustum.up.astype(np.float32)

            self.gaussians_gl.xyz = self.gaussian_cur.get_xyz.detach().cpu().numpy()

            self.gaussians_gl.opacity = (

                self.gaussian_cur.get_opacity.detach().cpu().numpy()

            )

            # 直接获取当前球谐特征和旋转

            self.gaussians_gl.sh = self.gaussian_cur.get_features.detach().cpu().numpy()[:, 0, :]

            self.gaussians_gl.rot = self.gaussian_cur.get_rotation.detach().cpu().numpy()

            # ========================================================

            # 【核心修复】：强制按 2DGS 处理，防止内存错位和精度溢出

            # ========================================================

            raw_scales = self.gaussian_cur.get_scaling.detach().cpu().numpy()

            # 无论如何，强制提取前两维 (防止 raw_scales 已经是乱的 N,3 甚至 N,4)

            scales_2d = raw_scales[:, :2]

            # 绝不能用 1e-6！改用 1e-3 (0.001)，既能保持椭圆扁平的外观，

            # 又能完美避开 OpenGL Shader 中 float32 计算协方差逆矩阵时的爆炸问题。

            z_scale = np.ones((scales_2d.shape[0], 1)) * 1e-3

            self.gaussians_gl.scale = np.concatenate([scales_2d, z_scale], axis=-1).astype(np.float32)

            # ========================================================

            self.update_activated_renderer_state(self.gaussians_gl)

            self.g_renderer.sort_and_update(self.g_camera)

            width, height = glfw.get_framebuffer_size(self.window_gl)

            self.g_renderer.draw()

            bufferdata = gl.glReadPixels(

                0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE

            )

            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)

            cv2.flip(img, 0, img)

            render_img = o3d.geometry.Image(img)

            glfw.swap_buffers(self.window_gl)
        else:
            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            render_img = o3d.geometry.Image(rgb)
        return render_img

    def render_gui(self):
        if not self.init:
            return
        current_cam = self.get_current_cam()

        ts_now = time.perf_counter()
        time_lapsed = ts_now - self.ts_prev
        if time_lapsed < (1.0 / self.render_fps):
            return
        self.ts_prev = ts_now

        results, deformation_data = self.rasterize(current_cam)
        if results is None:
            return
        self.render_img = self.render_o3d_image(results, current_cam, deformation_data)
        self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)

        if self.trj_replay_chbox.checked:
            self._on_trj_replay_chbox(True)
        if self.is_recording:
            self.screenrecord()

    def scene_update(self):
        self.receive_data(self.q_main2vis)
        self.render_gui()

    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                Log("Closing Visualization", tag="GUI")
                return

            def update():
                if self.step % 3 == 0:
                    self.scene_update()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()


if __name__ == "__main__":
    main()
