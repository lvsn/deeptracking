from deeptracking.data.dataset_utils import angle_distance, compute_axis
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.data.dataset import Dataset
from deeptracking.tracker.deeptracker import DeepTracker
import sys
import json
import time
import cv2
import numpy as np

from deeptracking.utils.data_logger import DataLogger
import os

from test_sensor import draw_debug

ESCAPE_KEY = 27


def log_pose_difference(prediction, ground_truth, logger):
    prediction_params = prediction.inverse().to_parameters(isDegree=True)
    ground_truth_params = ground_truth.inverse().to_parameters(isDegree=True)
    difference = np.zeros(6)
    for j in range(3):
        difference[j] = abs(prediction_params[j] - ground_truth_params[j])
        difference[j + 3] = abs(angle_distance(prediction_params[j + 3], ground_truth_params[j + 3]))
    logger.add_row(logger.get_dataframes_id()[0], difference)


if __name__ == '__main__':

    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    # Populate important data from config file
    OUTPUT_PATH = data["output_path"]
    VIDEO_PATH = data["video_path"]
    MODEL_PATH = data["model_path"]
    model_split_path = MODEL_PATH.split(os.sep)
    model_name = model_split_path[-1]
    model_folder = os.sep.join(model_split_path[:-1])
    MODELS_3D = data["models"]
    SHADER_PATH = data["shader_path"]
    CLOSED_LOOP_ITERATION = int(data["closed_loop_iteration"])
    SAVE_VIDEO = data["save_video"] == "True"
    SAVE_FRAMES = data["save_frames"] == "True"
    SHOW_AXIS = data["show_axis"] == "True"

    OBJECT_WIDTH = int(MODELS_3D[0]["object_width"])
    MODEL_3D_PATH = MODELS_3D[0]["model_path"]
    try:
        MODEL_3D_AO_PATH = MODELS_3D[0]["ambiant_occlusion_model"]
    except KeyError:
        MODEL_3D_AO_PATH = None
    USE_SENSOR = data["use_sensor"] == "True"
    RESET_FREQUENCY = int(data["reset_frequency"])
    frame_download_path = None


    video_data = Dataset(VIDEO_PATH)
    if not video_data.load():
        print("[ERROR] Error while loading video...")
        sys.exit(-1)
    frame_download_path = video_data.path
    # Makes the list a generator for compatibility with sensor's generator
    gen = lambda alist: [(yield i) for i in alist]
    frame_generator = gen(video_data.data_pose)
    camera = video_data.camera
    detection_mode = False
    debug_info = None

    tracker = DeepTracker(camera, data["model_file"], OBJECT_WIDTH)
    tracker.load(MODEL_PATH, MODEL_3D_PATH, MODEL_3D_AO_PATH, SHADER_PATH)
    tracker.print()
    # Frames from the generator are in camera coordinate
    previous_frame, previous_pose = next(frame_generator)
    previous_rgb, previous_depth = previous_frame.get_rgb_depth(frame_download_path)

    log_folder = os.path.join(model_folder, "scores")
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(log_folder, "video.avi"), fourcc, 30.0, (camera.width, camera.height))

    data_logger = DataLogger()
    data_logger.create_dataframe("{}_eval".format(model_name), ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz"))
    for i, (current_frame, ground_truth_pose) in enumerate(frame_generator):
        # get actual frame
        current_rgb, current_depth = current_frame.get_rgb_depth(frame_download_path)

        screen = current_rgb.copy()
        if RESET_FREQUENCY != 0 and i % RESET_FREQUENCY == 0:
            previous_pose = ground_truth_pose
        else:
            # process pose estimation of current frame given last pose
            start_time = time.time()
            if detection_mode:
                previous_pose = ground_truth_pose
            else:
                for j in range(CLOSED_LOOP_ITERATION):
                    predicted_pose, debug_info = tracker.estimate_current_pose(previous_pose, current_rgb, current_depth, debug=args.verbose)
                    previous_pose = predicted_pose
            print("[{}]Estimation processing time : {}".format(i, time.time() - start_time))
            if not USE_SENSOR:
                log_pose_difference(predicted_pose.inverse(), ground_truth_pose.inverse(), data_logger)
        if SHOW_AXIS:
            debug_info = None
        draw_debug(screen, previous_pose, ground_truth_pose, tracker, 1, debug_info)
        previous_rgb = current_rgb

        cv2.imshow("Debug", screen[:, :, ::-1])
        if SAVE_VIDEO:
            out.write(screen[:, :, ::-1])
        if SAVE_FRAMES:
            frame_folder = os.path.join(log_folder, "frames")
            if not os.path.exists(frame_folder):
                os.mkdir(frame_folder)
            cv2.imwrite(os.path.join(frame_folder, "{}.jpg".format(i)), screen[:, :, ::-1])
        key = cv2.waitKey(1)
        key_chr = chr(key & 255)
        if key != -1:
            print("pressed key id : {}, char : [{}]".format(key, key_chr))
        if key_chr == " ":
            print("Reset at frame : {}".format(i))
            previous_pose = ground_truth_pose
            detection_mode = not detection_mode
        if key == ESCAPE_KEY:
            break
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    data_logger.save(log_folder)
    if SAVE_VIDEO:
        out.release()

