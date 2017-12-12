import abc


class SensorBase:
    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def intrinsics(self):
        """
        return CameraParameters
        :return:
        """
        pass

    @abc.abstractmethod
    def get_frame(self, block=True):
        pass