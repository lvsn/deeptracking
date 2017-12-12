"""
    Camera's parameters and operations

    date : 2016-03-21
"""

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import numpy as np
import json
import os


class Camera:
    def __init__(self, focal, centers, size, distortion=np.array([[0., 0., 0., 0., 0.]])):
        self.focal_x = float(focal[0])
        self.focal_y = float(focal[1])
        self.center_x = float(centers[0])
        self.center_y = float(centers[1])
        self.width = int(size[0])
        self.height = int(size[1])
        self.distortion = distortion

    def project_points(self, points):
        computed_pixels = np.zeros((points.shape[0], 2))
        computed_pixels[:, 1] = points[:, 0] * self.focal_x / points[:, 2] + self.center_x
        computed_pixels[:, 0] = points[:, 1] * self.focal_y / points[:, 2] + self.center_y
        return np.round(computed_pixels)

    def backproject_depth(self, depth):
        constant_x = 1.0 / self.focal_x
        constant_y = 1.0 / self.focal_y
        row, col = depth.shape
        coords = np.zeros((row, col, 2), dtype=np.uint)
        coords[..., 0] = np.arange(row)[:, None]
        coords[..., 1] = np.arange(col)
        coords = coords.reshape((-1, 2))
        output = np.zeros((len(coords), 3))
        values = depth[coords[:, 0], coords[:, 1]]
        output[:, 0] = (coords[:, 1] - self.center_x) * values * constant_x
        output[:, 1] = (coords[:, 0] - self.center_y) * values * constant_y
        output[:, 2] = values
        return output

    def backproject_value(self, u, v, z):
        constant_x = 1.0 / self.focal_x
        constant_y = 1.0 / self.focal_y
        output = np.zeros((1, 3))
        output[:, 0] = (u - self.center_x) * z * constant_x
        output[:, 1] = (v - self.center_y) * z * constant_y
        output[:, 2] = z
        return output

    def set_ratio(self, ratio):
        self.focal_x = float(self.focal_x / ratio)
        self.focal_y = float(self.focal_y / ratio)
        self.center_x = float(self.center_x / ratio)
        self.center_y = float(self.center_y / ratio)
        self.width = int(self.width / ratio)
        self.height = int(self.height / ratio)

    def copy(self):
        return Camera((self.focal_x, self.focal_y),
                      (self.center_x, self.center_y),
                      (self.width, self.height),
                      self.distortion)

    def matrix(self):
        return np.array([[self.focal_x, 0, self.center_x],
                         [0, self.focal_y, self.center_y],
                         [0, 0, 1]])

    @staticmethod
    def from_matrix(matrix, width, height, distortion=np.array([[0., 0., 0., 0., 0.]])):
        focal_x = matrix[0, 0]
        focal_y = matrix[1, 1]

        center_x = matrix[0, 2]
        center_y = matrix[1, 2]

        return Camera((focal_x, focal_y), (center_x, center_y), (width, height), distortion)

    @staticmethod
    def load_from_json(path):
        file = path
        if path[-5:] != ".json":
            file = os.path.join(path, "camera.json")
        with open(file) as data_file:
            data = json.load(data_file)
        distortion = np.array([[0., 0., 0., 0., 0.]])
        try:
            distortion = np.array([data["distortion"]])
        except KeyError:
            pass
        camera = Camera((data["focalX"], data["focalY"]),
                                  (data["centerX"], data["centerY"]),
                                  (data["width"], data["height"]),
                                  distortion)
        return camera

    def save(self, path):
        dict = {
            "focalX": self.focal_x,
            "focalY": self.focal_y,
            "centerX": self.center_x,
            "centerY": self.center_y,
            "width": self.width,
            "height": self.height,
            "distortion": list(self.distortion[0])
        }
        with open(os.path.join(path, "camera.json"), 'w') as data_file:
            json.dump(dict, data_file)

    def __str__(self):
        ret = ""
        ret += "Focal x:" + str(self.focal_x) + "\n"
        ret += "Focal y:" + str(self.focal_y) + "\n"
        ret += "Center x:" + str(self.center_x) + "\n"
        ret += "Center y:" + str(self.center_y) + "\n"
        ret += "Width:" + str(self.width) + "\n"
        ret += "Height:" + str(self.height) + "\n"
        ret += "Distortion: " + str(self.distortion) + "\n"
        return ret
