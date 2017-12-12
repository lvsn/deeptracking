import abc


class TrackerBase:
    @abc.abstractmethod
    def estimate_current_pose(self, previous_pose, current_rgb, current_depth):
        pass

    @abc.abstractmethod
    def get_debug_screen(self, previous_frame):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def print(self):
        pass