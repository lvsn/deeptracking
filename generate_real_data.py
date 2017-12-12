from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.utils.transform import Transform
from deeptracking.data.dataset import Dataset
from deeptracking.data.dataset_utils import combine_view_transform, center_pixel, show_frames, compute_2Dboundingbox
from deeptracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deeptracking.data.dataset_utils import normalize_scale
from deeptracking.utils.camera import Camera
from scipy import ndimage
import sys
import json
import os
import math
import cv2
import numpy as np
import random

ESCAPE_KEY = 1048603


def mask_real_image(color, depth, depth_render):
    mask = (depth_render != 0).astype(np.uint8)[:, :, np.newaxis]
    masked_rgb = color * mask

    masked_hsv = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2HSV)
    saturation_mask = (masked_hsv[:, :, 2] <= SATURATION_THRESHOLD)[:, :, np.newaxis].astype(np.uint8)
    total_mask = np.bitwise_and(mask, saturation_mask)

    masked_color = color * total_mask
    masked_depth = depth[:total_mask.shape[0], :total_mask.shape[1]] * total_mask[:, :, 0]
    return masked_color, masked_depth


def random_z_rotation(rgb, depth, pose, camera):
    rotation = random.uniform(-180, 180)
    rotation_matrix = Transform()
    rotation_matrix.set_rotation(0, 0, math.radians(rotation))

    pixel = center_pixel(pose, camera)
    new_rgb = rotate_image(rgb, rotation, pixel[0])
    new_depth = rotate_image(depth, rotation, pixel[0])
    # treshold below 50 means we remove some interpolation noise, which cover small holes
    mask = (new_depth >= 50).astype(np.uint8)[:, :, np.newaxis]
    rgb_mask = np.all(new_rgb != 0, axis=2).astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    # erode rest of interpolation noise which will affect negatively future blendings
    eroded_mask = cv2.erode(mask, kernel, iterations=2)
    eroded_rgb_mask = cv2.erode(rgb_mask, kernel, iterations=2)
    new_depth = new_depth * eroded_mask
    new_rgb = new_rgb * eroded_rgb_mask[:, :, np.newaxis]
    new_pose = combine_view_transform(pose, rotation_matrix)
    return new_rgb, new_depth, new_pose


def rotate_image(img, angle, pivot):
    pivot = pivot.astype(np.int32)
    # double size of image while centering object
    pads = [[img.shape[0] - pivot[0], pivot[0]], [img.shape[1] - pivot[1], pivot[1]]]
    if len(img.shape) > 2:
        pads.append([0, 0])
    imgP = np.pad(img, pads, 'constant')
    # reduce size of matrix to rotate around the object
    if len(img.shape) > 2:
        total_y = np.sum(imgP.any(axis=(0, 2))) * 2.4
        total_x = np.sum(imgP.any(axis=(1, 2))) * 2.4
    else:
        total_y = np.sum(imgP.any(axis=0)) * 2.4
        total_x = np.sum(imgP.any(axis=1)) * 2.4
    cropy = int((imgP.shape[0] - total_y)/2)
    cropx = int((imgP.shape[1] - total_x)/2)
    imgP[cropy:-cropy, cropx:-cropx] = ndimage.rotate(imgP[cropy:-cropy, cropx:-cropx], angle,
                                                      reshape=False, prefilter=False)

    return imgP[pads[0][0]: -pads[0][1], pads[1][0]: -pads[1][1]]

if __name__ == '__main__':

    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    # Populate important data from config file
    MODELS = data["models"]
    SHADER_PATH = data["shader_path"]
    REAL_PATH = data["real_path"]
    OUTPUT_PATH = data["output_path"]
    SAMPLE_QUANTITY = int(data["sample_quantity"])
    TRANSLATION_RANGE = float(data["translation_range"])
    ROTATION_RANGE = math.radians(float(data["rotation_range"]))
    SPHERE_MIN_RADIUS = float(data["sphere_min_radius"])
    SPHERE_MAX_RADIUS = float(data["sphere_max_radius"])
    IMAGE_SIZE = (int(data["image_size"]), int(data["image_size"]))
    PRELOAD = data["preload"] == "True"
    SATURATION_THRESHOLD = int(data["saturation_threshold"])

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    real_dataset = Dataset(REAL_PATH)
    real_dataset.load()
    camera = Camera.load_from_json(real_dataset.path)
    real_dataset.camera = camera
    output_dataset = Dataset(OUTPUT_PATH, frame_class=data["save_type"])
    output_dataset.camera = camera
    window_size = (real_dataset.camera.width, real_dataset.camera.height)
    window = InitOpenGL(*window_size)

    model = MODELS[0]
    vpRender = ModelRenderer(model["model_path"], SHADER_PATH, real_dataset.camera, window, window_size)
    vpRender.load_ambiant_occlusion_map(model["ambiant_occlusion_model"])
    OBJECT_WIDTH = int(model["object_width"])

    metadata = {}
    metadata["translation_range"] = str(TRANSLATION_RANGE)
    metadata["rotation_range"] = str(ROTATION_RANGE)
    metadata["image_size"] = str(IMAGE_SIZE[0])
    metadata["save_type"] = data["save_type"]
    metadata["object_width"] = {}
    for model in MODELS:
        metadata["object_width"][model["name"]] = str(model["object_width"])
    metadata["min_radius"] = str(SPHERE_MIN_RADIUS)
    metadata["max_radius"] = str(SPHERE_MAX_RADIUS)
    for i in range(real_dataset.size()):
        frame, pose = real_dataset.data_pose[i]

        rgb_render, depth_render = vpRender.render(pose.transpose())
        rgb, depth = frame.get_rgb_depth(real_dataset.path)
        masked_rgb, masked_depth = mask_real_image(rgb, depth, depth_render)

        for j in range(SAMPLE_QUANTITY):
            rotated_rgb, rotated_depth, rotated_pose = random_z_rotation(masked_rgb, masked_depth, pose, real_dataset.camera)
            random_transform = Transform.random((-TRANSLATION_RANGE, TRANSLATION_RANGE),
                                                (-ROTATION_RANGE, ROTATION_RANGE))
            inverted_random_transform = Transform.from_parameters(*(-random_transform.to_parameters()))

            previous_pose = rotated_pose.copy()
            previous_pose = combine_view_transform(previous_pose, inverted_random_transform)

            rgbA, depthA = vpRender.render(previous_pose.transpose())
            bb = compute_2Dboundingbox(previous_pose, real_dataset.camera, OBJECT_WIDTH, scale=(1000, -1000, -1000))
            rgbA, depthA = normalize_scale(rgbA, depthA, bb, real_dataset.camera, IMAGE_SIZE)
            rgbB, depthB = normalize_scale(rotated_rgb, rotated_depth, bb, real_dataset.camera, IMAGE_SIZE)

            index = output_dataset.add_pose(rgbA, depthA, previous_pose)
            output_dataset.add_pair(rgbB, depthB, random_transform, index)
            iteration = i * SAMPLE_QUANTITY + j
            sys.stdout.write("Progress: %d%%   \r" % (int(iteration / (SAMPLE_QUANTITY * real_dataset.size()) * 100)))
            sys.stdout.flush()

            if iteration % 500 == 0:
                output_dataset.dump_images_on_disk()
            if iteration % 5000 == 0:
                output_dataset.save_json_files(metadata)

            if args.verbose:
                show_frames(rgbA, depthA, rgbB, depthB)
            cv2.imshow("testB", rgbB[:, :, ::-1])
            k = cv2.waitKey(1)
            if k == ESCAPE_KEY:
                break

    output_dataset.dump_images_on_disk()
    output_dataset.save_json_files(metadata)
