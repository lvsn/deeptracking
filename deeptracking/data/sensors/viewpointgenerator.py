from deeptracking.data.frame import Frame
import cv2

from deeptracking.utils.transform import Transform


class ViewpointGenerator:
    def __init__(self, sensor, detector):
        self.sensor = sensor
        self.detector = detector
        self.count = 0
        self.do_compute = False
        self.sensor.start()

    def __del__(self):
        self.sensor.stop()

    def compute_detection(self, do_compute):
        self.do_compute = do_compute

    def __next__(self):
        rgb, depth = self.sensor.get_frame()
        rgb = cv2.resize(rgb, (self.sensor.camera.width, self.sensor.camera.height))
        depth = cv2.resize(depth, (self.sensor.camera.width, self.sensor.camera.height))
        frame = Frame(rgb, depth, self.count)
        self.count += 1
        pose = None
        if self.do_compute:
            pose = self.detector.detect(rgb.copy())
        if pose is None:
            pose = Transform.from_parameters(0, 0, -1, 0, 0, 0)
        return frame, pose

    def __iter__(self):
        return self