"""
Utility to read rgbd files in folders ( used for background loading)
"""

import numpy as np
import os
import random
from PIL import Image


class RGBDDataset():
    def __init__(self, path, preload=False):
        self.do_preload = preload
        self.preloaded = []
        self.indexes = {}
        self.indexes_list = []
        self.path = path
        self.index_frames_()

    def index_frames_(self):
        dirs = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        for dir in dirs:
            dir_path = os.path.join(self.path, dir)
            files = [int(os.path.splitext(f)[0]) for f in os.listdir(dir_path) if os.path.splitext(f)[1] == ".png" and f[-5] != 'd']
            files.sort()
            files = [str(f) for f in files]
            self.indexes[dir] = files
            for file in files:
                self.indexes_list.append((dir, file))
                if self.do_preload:
                    color, depth = self.load_sample(dir, file)
                    self.preloaded.append((color, depth))

    def load_sample(self, dir, img):
        directory = os.path.join(self.path, dir)
        color = np.array(Image.open(os.path.join(directory, img + ".png")))
        depth = np.array(Image.open(os.path.join(directory, img + "d.png"))).astype(np.uint16)
        return color, depth

    def load_random_sample(self):
        rand_int = random.randint(0, len(self.indexes_list) - 1)
        if self.do_preload:
            color, depth = self.preloaded[rand_int]
        else:
            dir, file = self.indexes_list[rand_int]
            color, depth = self.load_sample(dir, file)
        return color, depth

    def load_random_image(self, size):
        color, depth = self.load_random_sample()
        x, y = RGBDDataset.get_random_crop(color.shape[0], color.shape[1], size)
        color = color[x:x+size, y:y+size, :]
        depth = depth[x:x+size, y:y+size]
        return color, depth

    def load_random_sequence(self):
        dir = random.choice(list(self.indexes.keys()))
        sequence = []
        for i in range(len(self.indexes[dir])):
            sequence.append(self.load_sample(dir, str(i)))
        return sequence

    @staticmethod
    def get_random_crop(w, h, size):
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return x, y

