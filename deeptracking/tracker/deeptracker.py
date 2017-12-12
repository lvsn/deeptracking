from deeptracking.tracker.trackerbase import TrackerBase
from deeptracking.utils.transform import Transform
from deeptracking.data.dataset_utils import combine_view_transform, normalize_depth, show_frames, compute_2Dboundingbox
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.data.dataset_utils import normalize_scale, normalize_channels, unnormalize_label
import PyTorchHelpers
import time
import numpy as np
import cv2


class DeepTracker(TrackerBase):
    def __init__(self, camera, model_path, object_width=0):
        self.image_size = None
        self.tracker_model = None
        self.translation_range = None
        self.rotation_range = None
        self.mean = None
        self.std = None
        self.debug_rgb = None
        self.debug_background = None
        self.camera = camera
        self.object_width = object_width

        # setup model
        model_class = PyTorchHelpers.load_lua_class(model_path, 'RGBDTracker')
        self.tracker_model = model_class('cuda', 'adam', 1)

        self.input_buffer = None
        self.prior_buffer = None

    def setup_renderer(self, model_3d_path, model_3d_ao_path, shader_path):
        window = InitOpenGL(*self.image_size)
        self.renderer = ModelRenderer(model_3d_path, shader_path, self.camera, window, self.image_size)
        if model_3d_ao_path is not None:
            self.renderer.load_ambiant_occlusion_map(model_3d_ao_path)

    def load(self, path, model_3d_path="", model_3d_ao_path="", shader_path=""):
        self.tracker_model.load(path)
        self.load_parameters_from_model_()
        if model_3d_path != "" and model_3d_ao_path != "" and shader_path != "":
            self.setup_renderer(model_3d_path, model_3d_ao_path, shader_path)

    def print(self):
        print(self.tracker_model.model_string())

    def load_parameters_from_model_(self):
        self.image_size = (
        int(self.tracker_model.get_configs("input_size")), int(self.tracker_model.get_configs("input_size")))
        self.translation_range = float(self.tracker_model.get_configs("translation_range"))
        self.rotation_range = float(self.tracker_model.get_configs("rotation_range"))
        self.input_buffer = np.ndarray((1, 8, self.image_size[0], self.image_size[1]), dtype=np.float32)
        self.prior_buffer = np.ndarray((1, 7), dtype=np.float32)
        self.mean = self.tracker_model.get_configs("mean_matrix").asNumpyTensor()
        self.std = self.tracker_model.get_configs("std_matrix").asNumpyTensor()

    def set_configs_(self, configs):
        self.tracker_model.set_configs(configs)

    def compute_render(self, previous_pose, bb):
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])
        self.renderer.setup_camera(self.camera, left, right, bottom, top)
        render_rgb, render_depth = self.renderer.render(previous_pose.transpose())
        return render_rgb, render_depth

    def estimate_current_pose(self, previous_pose, current_rgb, current_depth, debug=False, debug_time=False):
        if debug_time:
            start_time = time.time()
        bb = compute_2Dboundingbox(previous_pose, self.camera, self.object_width, scale=(1000, 1000, -1000))
        bb2 = compute_2Dboundingbox(previous_pose, self.camera, self.object_width, scale=(1000, -1000, -1000))
        if debug_time:
            print("Compute BB : {}".format(time.time() - start_time))
            start_time = time.time()
        rgbA, depthA = self.compute_render(previous_pose, bb)
        if debug_time:
            print("Render : {}".format(time.time() - start_time))
            start_time = time.time()
        rgbB, depthB = normalize_scale(current_rgb, current_depth, bb2, self.camera, self.image_size)
        debug_info = (rgbA, bb2, np.hstack((rgbA, rgbB)))

        rgbA = rgbA.astype(np.float)
        rgbB = rgbB.astype(np.float)
        depthA = depthA.astype(np.float)
        depthB = depthB.astype(np.float)

        depthA = normalize_depth(depthA, previous_pose)
        depthB = normalize_depth(depthB, previous_pose)

        if debug:
            show_frames(rgbA, depthA, rgbB, depthB)
        rgbA, depthA = normalize_channels(rgbA, depthA, self.mean[:4], self.std[:4])
        rgbB, depthB = normalize_channels(rgbB, depthB, self.mean[4:], self.std[4:])

        self.input_buffer[0, 0:3, :, :] = rgbA
        self.input_buffer[0, 3, :, :] = depthA
        self.input_buffer[0, 4:7, :, :] = rgbB
        self.input_buffer[0, 7, :, :] = depthB
        self.prior_buffer[0] = np.array(previous_pose.to_parameters(isQuaternion=True))
        if debug_time:
            print("Normalize : {}".format(time.time() - start_time))
            start_time = time.time()
        prediction = self.tracker_model.test([self.input_buffer, self.prior_buffer]).asNumpyTensor()
        if debug_time:
            print("Network time : {}".format(time.time() - start_time))
        prediction = unnormalize_label(prediction, self.translation_range, self.rotation_range)
        if debug:
            print("Prediction : {}".format(prediction))
        prediction = Transform.from_parameters(*prediction[0], is_degree=True)
        current_pose = combine_view_transform(previous_pose, prediction)
        return current_pose, debug_info

