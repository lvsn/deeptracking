from deeptracking.data.sensors.sensorbase import SensorBase
from deeptracking.utils.camera import Camera
import pyrealsense as pyrs


class Realsense(SensorBase):
    def __init__(self):
        self.device = None

    def start(self):
        pyrs.start()
        self.device = pyrs.Device(device_id=0)

    def stop(self):
        pyrs.stop()
        self.device = None

    def intrinsics(self):
        distortion = []
        for i in range(5):
            distortion.append(self.device.colour_intrinsics.coeffs[i])
        camera = Camera((self.device.colour_intrinsics.fx, self.device.colour_intrinsics.fy),
               (self.device.colour_intrinsics.ppx, self.device.colour_intrinsics.ppy),
               (self.device.colour_intrinsics.width, self.device.colour_intrinsics.height),
               distortion)
        print(camera)
        return camera

    def get_frame(self, block=True):
        # TODO implement non blocking operation
        self.device.wait_for_frame()
        return self.device.colour, self.device.dac
