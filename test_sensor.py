import cv2
import sys
import json
import time
import numpy as np
import os

from deeptracking.data.dataset_utils import compute_axis, image_blend
from deeptracking.data.sensors.kinect2 import Kinect2
from deeptracking.data.sensors.viewpointgenerator import ViewpointGenerator
from deeptracking.detector.detector_aruco import ArucoDetector
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.tracker.deeptracker import DeepTracker
from deeptracking.utils.filters import MeanFilter

ESCAPE_KEY = 1048603
SPACE_KEY = 1048608
UNITY_DEMO = False
DEBUG_TIME = False
DEBUG = True


def draw_debug(img, pose, gt_pose, tracker, alpha, debug_info):
    if debug_info is not None:
        img_render, bb, _ = debug_info
        img_render = cv2.resize(img_render, (bb[2, 1] - bb[0, 1], bb[1, 0] - bb[0, 0]))
        crop = img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :]
        h, w, c = crop.shape
        blend = image_blend(img_render[:h, :w, ::-1], crop)
        img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :] = cv2.addWeighted(img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :],
                                                                       1 - alpha, blend, alpha, 1)
    else:
        axis = compute_axis(pose, tracker.camera, tracker.object_width, scale=(1000, -1000, -1000))
        axis_gt = compute_axis(gt_pose, tracker.camera, tracker.object_width, scale=(1000, -1000, -1000))

        cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[1, ::-1]), (0, 0, 155), 3)
        cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[2, ::-1]), (0, 155, 0), 3)
        cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[3, ::-1]), (155, 0, 0), 3)

        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), (0, 0, 255), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), (0, 255, 0), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), (255, 0, 0), 3)

alpha = 1
def trackbar(x):
    global alpha
    alpha = x/100

if __name__ == '__main__':

    if UNITY_DEMO:
        TCP_IP = "0.0.0.0"
        TCP_PORT = 9050
        print("Activating Unity server on {}:{}".format(TCP_IP, TCP_PORT))
        import pyunity.server as server
        from pyunity.frame import ExampleMetadata

        meta = ExampleMetadata()
        unity_server = server.Server(TCP_IP, TCP_PORT)
        while not unity_server.has_connection():
            time.sleep(1)
        output_rot_filter = MeanFilter(2)
        output_trans_filter = MeanFilter(2)

    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    # Populate important data from config file
    OUTPUT_PATH = data["output_path"]
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    MODEL_PATH = data["model_path"]
    model_split_path = MODEL_PATH.split(os.sep)
    model_name = model_split_path[-1]
    model_folder = os.sep.join(model_split_path[:-1])
    MODELS_3D = data["models"]
    SHADER_PATH = data["shader_path"]
    CLOSED_LOOP_ITERATION = int(data["closed_loop_iteration"])
    SAVE_VIDEO = data["save_video"] == "True"
    SHOW_DEPTH = data["show_depth"] == "True"
    SHOW_ZOOM = data["show_zoom"] == "True"

    OBJECT_WIDTH = int(MODELS_3D[0]["object_width"])
    MODEL_3D_PATH = MODELS_3D[0]["model_path"]
    try:
        MODEL_3D_AO_PATH = MODELS_3D[0]["ambiant_occlusion_model"]
    except KeyError:
        MODEL_3D_AO_PATH = None

    sensor = Kinect2(data["sensor_camera_path"])
    detector = ArucoDetector(sensor.camera, data["detector_layout_path"])
    frame_generator = ViewpointGenerator(sensor, detector)
    camera = sensor.camera
    detection_mode = True
    frame_generator.compute_detection(detection_mode)

    tracker = DeepTracker(camera, data["model_file"], OBJECT_WIDTH)
    tracker.load(MODEL_PATH, MODEL_3D_PATH, MODEL_3D_AO_PATH, SHADER_PATH)
    tracker.print()
    # Frames from the generator are in camera coordinate
    previous_frame, previous_pose = next(frame_generator)
    previous_rgb, previous_depth = previous_frame.get_rgb_depth(None)

    cv2.namedWindow('image')
    cv2.createTrackbar('transparency', 'image', 0, 100, trackbar)

    out = None
    debug_info = None
    for i, (current_frame, ground_truth_pose) in enumerate(frame_generator):
        # get actual frame
        if DEBUG_TIME:
            start_time = time.time()

        current_rgb, current_depth = current_frame.get_rgb_depth(None)
        screen_rgb = current_rgb.copy()
        if SHOW_DEPTH:
            screen_depth = (current_depth / np.max(current_depth) * 255).astype(np.uint8)[:, :, np.newaxis]
            screen_depth = np.repeat(screen_depth, 3, axis=2)

        if detection_mode:
            previous_pose = ground_truth_pose
        else:
            for j in range(CLOSED_LOOP_ITERATION):
                predicted_pose, debug_info = tracker.estimate_current_pose(previous_pose, current_rgb, current_depth,
                                                                           debug=args.verbose,
                                                                           debug_time=DEBUG_TIME)
                previous_pose = predicted_pose

        draw_debug(screen_rgb, previous_pose, ground_truth_pose, tracker, alpha, debug_info)
        if SHOW_DEPTH:
            draw_debug(screen_depth, previous_pose, ground_truth_pose, tracker, alpha, debug_info)
        previous_rgb = current_rgb
        if UNITY_DEMO:
            if meta.camera_parameters is None:
                meta.camera_parameters = camera.copy()
                meta.camera_parameters.distortion = meta.camera_parameters.distortion.tolist()
            meta.object_pose = []
            if previous_pose:
                params = previous_pose.to_parameters()
                meta.add_object_pose(*params)
            unity_server.send_data_to_clients(current_rgb[:, :, ::-1], meta)
        if DEBUG:
            min_x = 80
            max_x = -150
            screen = screen_rgb[:, min_x:max_x, :]
            if SHOW_DEPTH:
                screen = np.concatenate((screen_rgb[:, min_x:max_x, :], screen_depth[:, min_x:max_x, :]), axis=1)
            if SHOW_ZOOM and debug_info is not None:
                _, _, zoom = debug_info
                zoom_h, zoom_w, zoom_c = zoom.shape
                screen[:zoom_h + 6, :zoom_w + 6, :] = 255
                screen[3:zoom_h + 3, 3:zoom_w + 3, :] = zoom
            cv2.imshow("image", screen[:, :, ::-1])
            if SAVE_VIDEO:
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(os.path.join(OUTPUT_PATH, "video.avi"), fourcc, 12.0,
                                          (screen.shape[1], screen.shape[0]))
                out.write(screen[:, :, ::-1])
            key = cv2.waitKey(1)
            key_chr = chr(key & 255)
            if key != -1:
                print("pressed key id : {}, char : [{}]".format(key, key_chr))
            if key == SPACE_KEY:
                print("Reset at frame : {}".format(i))
                previous_pose = ground_truth_pose
                detection_mode = not detection_mode
                frame_generator.compute_detection(detection_mode)
            if key == ESCAPE_KEY:
                break
        if DEBUG_TIME:
            print("[{}]Estimation processing time : {}".format(i, time.time() - start_time))

    if SAVE_VIDEO:
        out.release()

