"""
    Keeps a reference to a rgb and depth images (given an id). keeps data on disk or ram depending on need

    date : 2017-23-03
"""

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import numpngw
import os
import numpy as np
from PIL import Image


class Frame:
    def __init__(self, rgb, depth, id):
        self.rgb = rgb
        self.depth = depth
        self.id = id

    def is_on_disk(self):
        return self.rgb is None and self.depth is None

    def exists(self, path):
        return os.path.exists(os.path.join(path, '{}.png').format(self.id)) and \
               os.path.exists(os.path.join(path, '{}d.png').format(self.id))

    def get_rgb_depth(self, path, keep_in_ram=False):
        """
        getter to images, if keep in ram is false, the reference wont be kept by this object
        :param path:
        :param keep_in_ram:
        :return:
        """
        if self.is_on_disk():
            self.load(path)
        rgb = self.rgb
        depth = self.depth
        if not keep_in_ram:
            self.clear_image()
        return rgb, depth

    def clear_image(self):
        self.rgb = None
        self.depth = None

    def dump(self, path):
        if not self.is_on_disk():
            numpngw.write_png(os.path.join(path, '{}.png').format(self.id), self.rgb)
            numpngw.write_png(os.path.join(path, '{}d.png').format(self.id), self.depth.astype(np.uint16))
            self.clear_image()

    def load(self, path):
        self.rgb = np.array(Image.open(os.path.join(path, self.id + ".png")))
        self.depth = np.array(Image.open(os.path.join(path, self.id + "d.png"))).astype(np.uint16)


class FrameNumpy(Frame):
    def __init__(self, rgb, depth, id):
        super().__init__(rgb, depth, id)

    def dump(self, path):
        if not self.is_on_disk():
            depth8 = self.numpy_int16_to_uint8(self.depth)
            frame = np.concatenate((self.rgb, depth8), axis=2)
            np.save(os.path.join(path, self.id), frame)
            self.clear_image()

    def load(self, path):
        frame = np.load(os.path.join(path, "{}.npy".format(self.id)))
        self.depth = self.numpy_uint8_to_int16(frame[:, :, 3:])
        self.rgb = frame[:, :, 0:3]

    @staticmethod
    def numpy_int16_to_uint8(depth):
        x, y = depth.shape
        out = np.ndarray((x, y, 2), dtype=np.uint8)
        out[:, :, 0] = np.right_shift(depth, 8)
        out[:, :, 1] = depth.astype(np.uint8)
        return out

    @staticmethod
    def numpy_uint8_to_int16(depth8):
        x, y, c = depth8.shape
        out = np.ndarray((x, y), dtype=np.int16)
        out[:, :] = depth8[:, :, 0]
        out = np.left_shift(out, 8)
        out[:, :] += depth8[:, :, 1]
        return out
