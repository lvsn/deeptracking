"""
    use a pose detector (aruco, checkboard) and compute the pose on the whole dataset
"""

from deeptracking.data.dataset import Dataset
from deeptracking.data.dataset_utils import image_blend
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.utils.camera import Camera
from deeptracking.utils.transform import Transform
import cv2
import os
import numpy as np

from deeptracking.detector.detector_aruco import ArucoDetector

if __name__ == '__main__':
    dataset_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking/sequences/skull"
    detector_path = "../deeptracking/detector/aruco_layout.xml"
    model_path = "/home/mathieu/Dataset/3D_models/skull/skull.ply"
    model_ao_path = "/home/mathieu/Dataset/3D_models/skull/skull_ao.ply"
    shader_path = "../deeptracking/data/shaders"

    dataset = Dataset(dataset_path)
    offset = Transform.from_matrix(np.load(os.path.join(dataset.path, "offset.npy")))

    camera = Camera.load_from_json(dataset_path)
    dataset.camera = camera
    files = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[-1] == ".png" and 'd' not in os.path.splitext(f)[0]]
    detector = ArucoDetector(camera, detector_path)
    window = InitOpenGL(camera.width, camera.height)
    vpRender = ModelRenderer(model_path, shader_path, camera, window, (camera.width, camera.height))
    vpRender.load_ambiant_occlusion_map(model_ao_path)
    ground_truth_pose = None

    for i in range(len(files)):
        img = cv2.imread(os.path.join(dataset.path, "{}.png".format(i)))
        detection = detector.detect(img)
        if detection is not None:
            ground_truth_pose = detection
            ground_truth_pose.combine(offset.inverse(), copy=False)
        else:
            print("[WARN]: frame {} has not been detected.. using previous detection".format(i))
        dataset.add_pose(None, None, ground_truth_pose)
        rgb_render, depth_render = vpRender.render(ground_truth_pose.transpose())
        bgr_render = rgb_render[:, :, ::-1].copy()
        img = image_blend(bgr_render, img)

        cv2.imshow("view", img)
        cv2.waitKey(1)
    dataset.save_json_files({"save_type": "png"})
