from deeptracking.detector.detectorbase import DetectorBase
from deeptracking.utils.transform import Transform
import numpy as np
import cv2


class ChessboardDetector(DetectorBase):
    def __init__(self, camera, chess_shape=(5, 4), chess_size=26):
        self.camera = camera
        self.chess_shape = chess_shape
        self.obj_points = np.zeros((chess_shape[0] * chess_shape[1], 3))
        for i in range(chess_shape[1]):
            for j in range(chess_shape[0]):
                self.obj_points[i * chess_shape[0] + j] = [chess_size * j, chess_size * i, 0]

    def detect(self, img):
        if len(img.shape) > 2:
            raise Exception("ChessboardDetector uses gray image as input")
        detection = None
        ret, corners = cv2.findChessboardCorners(img, self.chess_shape, None)
        if ret:
            ret, rvec, tvec = cv2.solvePnP(self.obj_points, corners, self.camera.matrix(), np.array([0, 0, 0, 0, 0]))
            # invert axis convention
            rvec[1] = -rvec[1]
            rvec[2] = -rvec[2]
            tvec[1] = -tvec[1]
            tvec[2] = -tvec[2]
            detection = Transform()
            detection.matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
            detection.set_translation(tvec[0] / 1000, tvec[1] / 1000, tvec[2] / 1000)
        return detection