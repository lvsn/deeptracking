import numpy as np
import math
from deeptracking.utils.transform import Transform
from scipy.misc import imresize

try:
    import cv2
    import matplotlib.pyplot as plt
except ImportError:
    pass


def crop_viewpoint(viewpoint):
    viewpoint.frame.color = crop_image(viewpoint.frame.color)
    viewpoint.frame.depth = crop_image(viewpoint.frame.depth)


def crop_image(image, crop_center, size=100):
    y = int(crop_center[1] - size / 2)
    x = int(crop_center[0] - size / 2)
    image = image[y: y + size, x: x + size]
    return image


def unnormalize_label(params, max_translation, max_rotation_rad):
    params[:, :3] *= max_translation
    params[:, 3:] *= math.degrees(max_rotation_rad)
    return params


def angle_distance(unit1, unit2):
    phi = abs(unit2 - unit1) % 360
    sign = 1
    # used to calculate sign
    if not ((0 <= unit1 - unit2 <= 180) or (-180 >= unit1 - unit2 >= -360)):
        sign = -1
    if phi > 180:
        result = 360 - phi
    else:
        result = phi
    return result * sign


def combine_view_transform(vp, view_transform):
    """
    combines a camera space transform with a camera axis dependent transform.
    Whats important here is that view transform's translation represent the displacement from
    each axis, and rotation from each axis. The rotation is applied around the translation point of view_transform.
    :param vp:
    :param view_transform:
    :return:
    """
    camera_pose = vp.copy()
    R = camera_pose.rotation
    T = camera_pose.translation
    rand_R = view_transform.rotation
    rand_T = view_transform.translation

    rand_R.combine(R)
    T.combine(rand_R)
    rand_T.combine(T)
    return rand_T


def normalize_scale(color, depth, boundingbox, camera, output_size=(100, 100)):

    left = np.min(boundingbox[:, 1])
    right = np.max(boundingbox[:, 1])
    top = np.min(boundingbox[:, 0])
    bottom = np.max(boundingbox[:, 0])

    # Compute offset if bounding box goes out of the frame (0 padding)
    h, w, c = color.shape
    crop_w = right - left
    crop_h = bottom - top
    color_crop = np.zeros((crop_h, crop_w, 3), dtype=color.dtype)
    depth_crop = np.zeros((crop_h, crop_w), dtype=np.float)
    top_offset = abs(min(top, 0))
    bottom_offset = min(crop_h - (bottom - h), crop_h)
    right_offset = min(crop_w - (right - w), crop_w)
    left_offset = abs(min(left, 0))

    if top < 0:
        top = 0
    if left < 0:
        left = 0
    color_crop[top_offset:bottom_offset, left_offset:right_offset, :] = color[top:bottom, left:right, :]
    depth_crop[top_offset:bottom_offset, left_offset:right_offset] = depth[top:bottom, left:right]

    resized_rgb = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST)
    resized_depth = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST)

    mask_rgb = resized_rgb != 0
    mask_depth = resized_depth != 0
    resized_depth = resized_depth.astype(np.int16)
    final_rgb = resized_rgb * mask_rgb
    final_depth = resized_depth * mask_depth
    return final_rgb, final_depth


def cv_normalize_scale(color, depth, pose, camera, output_size=(100, 100), scale_size=230):
    pose = pose.inverse()
    pixels = compute_2Dboundingbox(pose, camera, scale_size)

    # pad zeros if the crop happens outside of original image
    lower_x = 0
    lower_y = 0
    higher_x = 0
    higher_y = 0
    if pixels[0, 0] < 0:
        lower_x = -pixels[0, 0]
        pixels[:, 0] += lower_x
    if pixels[0, 1] < 0:
        lower_y = -pixels[0, 1]
        pixels[:, 1] += lower_y
    if pixels[1, 0] > camera.width:
        higher_x = pixels[1, 0] - camera.width
    if pixels[1, 1] > camera.height:
        higher_y = pixels[1, 1] - camera.height

    color = np.pad(color, ((lower_y, higher_y), (lower_x, higher_x), (0, 0)), mode="constant", constant_values=0)
    depth = np.pad(depth, ((lower_y, higher_y), (lower_x, higher_x)), mode="constant", constant_values=0)
    color_crop = color[pixels[0, 0]:pixels[1, 0], pixels[0, 1]:pixels[2, 1], :]
    depth_crop = depth[pixels[0, 0]:pixels[1, 0], pixels[0, 1]:pixels[2, 1]].astype(np.float)
    mask_depth = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST) != 0
    mask_rgb = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST) != 0
    resized_depth_crop = cv2.resize(depth_crop, output_size, interpolation=cv2.INTER_NEAREST)
    resized_color_crop = cv2.resize(color_crop, output_size, interpolation=cv2.INTER_NEAREST)
    return resized_color_crop * mask_rgb, resized_depth_crop * mask_depth


def compute_2Dboundingbox(pose, camera, scale_size=230, scale=(1, 1, 1)):
    obj_x = pose.matrix[0, 3] * scale[0]
    obj_y = pose.matrix[1, 3] * scale[1]
    obj_z = pose.matrix[2, 3] * scale[2]
    offset = scale_size / 2
    points = np.ndarray((4, 3), dtype=np.float)
    points[0] = [obj_x - offset, obj_y - offset, obj_z]
    points[1] = [obj_x - offset, obj_y + offset, obj_z]
    points[2] = [obj_x + offset, obj_y - offset, obj_z]
    points[3] = [obj_x + offset, obj_y + offset, obj_z]
    return camera.project_points(points).astype(np.int32)


def compute_axis(pose, camera, scale_size, scale=(1, 1, 1)):
    points = np.ndarray((4, 3), dtype=np.float)
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2] = [0, 1, 0]
    points[3] = [0, 0, 1]
    points *= 0.1
    camera_points = pose.dot(points)
    camera_points[:, 0] *= -1
    return camera.project_points(camera_points).astype(np.int32)


def center_pixel(pose, camera):
    obj_x = pose.matrix[0, 3] * 1000
    obj_y = pose.matrix[1, 3] * 1000
    obj_z = pose.matrix[2, 3] * 1000
    point = [obj_x, -obj_y, -obj_z]
    return camera.project_points(np.array([point])).astype(np.uint32)


def image_blend(foreground, background):
    """
    Uses pixel 0 to compute blending mask
    :param foreground:
    :param background:
    :return:
    """
    if len(foreground.shape) == 2:
        mask = foreground[:, :] == 0
    else:
        mask = foreground[:, :, 0] == 0
        mask = mask[:, :, np.newaxis]
    return background * mask + foreground


def normalize_channels(rgb, depth, mean, std):
    """
    Normalize image by negating mean and dividing by std (precomputed)
    :param self:
    :param rgb:
    :param depth:
    :param type:
    :return:
    """
    rgb = rgb.T
    depth = depth.T
    rgb -= mean[:3, np.newaxis, np.newaxis]
    rgb /= std[:3, np.newaxis, np.newaxis]
    depth -= mean[3, np.newaxis, np.newaxis]
    depth /= std[3, np.newaxis, np.newaxis]
    return rgb, depth


def unormalize_channels(rgb, depth, mean, std):
    rgb = rgb.T
    depth = depth.T
    rgb *= std[np.newaxis, np.newaxis, :3]
    rgb += mean[np.newaxis, np.newaxis, :3]
    depth *= std[np.newaxis, np.newaxis, 3]
    depth += mean[np.newaxis, np.newaxis, 3]
    return rgb, depth


def show_frames(rgbA, depthA, rgbB, depthB):
    fig, axis = plt.subplots(2, 2)
    ax1, ax2 = axis[0, :]
    ax3, ax4 = axis[1, :]
    ax1.imshow(rgbA.astype(np.uint8))
    ax2.imshow(rgbB.astype(np.uint8))
    ax3.imshow(depthA)
    ax4.imshow(depthB)
    plt.show()


def show_frames_from_buffer(image_buffer, mean, std):
    rgbA = image_buffer[0, 0:3, :, :]
    depthA = image_buffer[0, 3, :, :]
    rgbB = image_buffer[0, 4:7, :, :]
    depthB = image_buffer[0, 7, :, :]
    rgbA, depthA = unormalize_channels(rgbA, depthA, mean[:4], std[:4])
    rgbB, depthB = unormalize_channels(rgbB, depthB, mean[4:], std[4:])
    show_frames(rgbA, depthA, rgbB, depthB)


def normalize_depth(depth, pose):
    depth = depth.astype(np.float32)
    zero_mask = depth == 0
    depth += pose.matrix[2, 3] * 1000
    depth[zero_mask] = 5000
    return depth