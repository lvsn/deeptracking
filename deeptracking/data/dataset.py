import os
import numpy as np
import json
import math

from deeptracking.data.parallelminibatch import ParallelMinibatch
from deeptracking.data.dataset_utils import normalize_channels, normalize_depth
from deeptracking.utils.transform import Transform
from deeptracking.utils.camera import Camera
from deeptracking.data.frame import Frame, FrameNumpy


class Dataset(ParallelMinibatch):
    def __init__(self, folder_path, frame_class="png", minibatch_size=64, max_parallel_buffer_size=0, max_samples=0):
        ParallelMinibatch.__init__(self, max_parallel_buffer_size)
        self.path = folder_path
        self.data_pose = []
        self.data_pair = {}
        self.metadata = {}
        self.camera = None
        self.frame_class = None
        self.set_save_type(frame_class)
        self.mean = None
        self.std = None
        self.data_augmentation = None
        self.minibatch_size = minibatch_size
        self.max_size = max_samples

    def set_save_type(self, frame_class):
        if frame_class == "numpy":
            self.frame_class = FrameNumpy
        else:
            self.frame_class = Frame

    def compute_mean_std(self):
        max_size = 10000
        indexes = self.compute_minibatches_permutations_()[:int(max_size/self.minibatch_size)]
        self.mean = self.compute_channels_mean(indexes)
        self.std = self.compute_channels_std(indexes, self.mean)
        np.save(os.path.join(self.path, "mean.npy"), self.mean)
        np.save(os.path.join(self.path, "std.npy"), self.std)

    def compute_channels_mean(self, batch_indexes):
        # todo could be done in parallel
        channel_means = np.zeros(8)
        processed_images = 0
        for index in batch_indexes:
            image_buffer, _, _ = self.load_minibatch(index)
            image_means = np.mean(image_buffer, axis=(2, 3))
            channel_means += np.sum(image_means, axis=0)
            processed_images += image_buffer.shape[0]
        channel_means = channel_means / processed_images
        return channel_means

    def compute_channels_std(self, batch_indexes, channel_mean):
        channel_std = np.zeros(8)
        processed_images = 0
        for index in batch_indexes:
            image_buffer, _, _ = self.load_minibatch(index)
            image_means = np.mean(image_buffer, axis=(2, 3))
            channel_std += np.sum(np.square(image_means - channel_mean), axis=0)
            processed_images += image_buffer.shape[0]
        channel_std = np.sqrt(channel_std / processed_images)
        return channel_std

    def add_pose(self, rgb, depth, pose):
        index = self.size()
        frame = self.frame_class(rgb, depth, str(index))
        self.data_pose.append((frame, pose))
        return index

    def pair_size(self, id):
        if id not in self.data_pair:
            return 0
        else:
            return len(self.data_pair[id])

    def add_pair(self, rgb, depth, pose, id):
        if id >= len(self.data_pose):
            raise IndexError("impossible to add pair if pose does not exists")
        if id in self.data_pair:
            frame = self.frame_class(rgb, depth, "{}n{}".format(id, len(self.data_pair[id]) - 1))
            self.data_pair[id].append((frame, pose))
        else:
            frame = self.frame_class(rgb, depth, "{}n0".format(id))
            self.data_pair[id] = [(frame, pose)]

    def dump_images_on_disk(self, verbose=False):
        """
        Unload all images data from ram and save them to the dataset's path ( can be reloaded with load_from_disk())
        :return:
        """
        if verbose:
            print("Save {} viewpoints".format(len(self.data_pose)))
        for frame, pose in self.data_pose:
            if verbose:
                print("Save frame {}".format(frame.id))
            if int(frame.id) in self.data_pair:
                for pair_frame, pair_pose in self.data_pair[int(frame.id)]:
                    pair_frame.dump(self.path)
            frame.dump(self.path)

    def save_json_files(self, metadata):
        viewpoints_data = {}
        for frame, pose in self.data_pose:
            self.insert_pose_in_dict(viewpoints_data, frame.id, pose)
            if int(frame.id) in self.data_pair:
                viewpoints_data[frame.id]["pairs"] = len(self.data_pair[int(frame.id)])
                for pair_frame, pair_pose in self.data_pair[int(frame.id)]:
                    self.insert_pose_in_dict(viewpoints_data, pair_frame.id, pair_pose)
            else:
                viewpoints_data[frame.id]["pairs"] = 0
        viewpoints_data["metaData"] = metadata
        with open(os.path.join(self.path, "viewpoints.json"), 'w') as outfile:
            json.dump(viewpoints_data, outfile)
        if self.camera is None:
            raise Exception("Camera is not defined for dataset...")
        self.camera.save(self.path)

    def load(self):
        """
        Load a viewpoints.json to dataset's structure
        Todo: datastructure should be more similar to json structure...
        :return: return false if the dataset is empty.
        """
        # Load viewpoints file and camera file
        try:
            with open(os.path.join(self.path, "viewpoints.json")) as data_file:
                data = json.load(data_file)
            self.camera = Camera.load_from_json(self.path)
        except FileNotFoundError:
            return False
        self.metadata = data["metaData"]
        self.set_save_type(self.metadata["save_type"])
        count = 0
        # todo this is not clean!i
        while True:
            try:
                id = str(count)
                pose = Transform.from_parameters(*[float(data[id]["vector"][str(x)]) for x in range(6)])
                self.add_pose(None, None, pose)
                if "pairs" in data[id]:
                    for i in range(int(data[id]["pairs"])):
                        pair_id = "{}n{}".format(id, i)
                        pair_pose = Transform.from_parameters(*[float(data[pair_id]["vector"][str(x)]) for x in range(6)])
                        self.add_pair(None, None, pair_pose, count)
                count += 1

            except KeyError:
                print("Keyerror {} at {}".format(id, count))
                break
        return True

    @staticmethod
    def insert_pose_in_dict(dict, key, item):
        params = {}
        for i, param in enumerate(item.to_parameters()):
            params[str(i)] = str(param)
        dict[key] = {"vector": params}

    def size(self):
        return len(self.data_pose)

    def set_data_augmentation(self, data_augmentation):
        self.data_augmentation = data_augmentation

    def get_image_pair(self, index):
        frame, pose = self.data_pose[index]
        frame_pair, pose_pair = self.data_pair[index][0]
        rgb, depth = frame.get_rgb_depth(self.path)
        rgb_pair, depth_pair = frame_pair.get_rgb_depth(self.path)
        return rgb, depth, pose, rgb_pair, depth_pair

    def load_image(self, index):
        frame, pose = self.data_pose[index]
        rgb, depth = frame.get_rgb_depth(self.path)
        return rgb, depth, pose

    def load_pair(self, index, pair_id):
        frame, pose = self.data_pair[index][pair_id]
        rgb, depth = frame.get_rgb_depth(self.path)
        return rgb, depth, pose

    def get_sample(self, index, image_buffer, prior_buffer, label_buffer, buffer_index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)
        if self.data_augmentation is not None:
            rgbA, depthA = self.data_augmentation.augment(rgbA, depthA, initial_pose, real=False)
            rgbB, depthB = self.data_augmentation.augment(rgbB, depthB, initial_pose, real=True)

        depthA = normalize_depth(depthA, initial_pose)
        depthB = normalize_depth(depthB, initial_pose)

        rgbA = rgbA.astype(np.float32)
        rgbB = rgbB.astype(np.float32)
        depthA = depthA.astype(np.float32)
        depthB = depthB.astype(np.float32)

        if self.mean is None or self.std is None:
            rgbA, rgbB = rgbA.T, rgbB.T
            depthA, depthB = depthA.T, depthB.T
        else:
            rgbA, depthA = normalize_channels(rgbA, depthA, self.mean[:4], self.std[:4])
            rgbB, depthB = normalize_channels(rgbB, depthB, self.mean[4:], self.std[4:])

        image_buffer[buffer_index, 0:3, :, :] = rgbA
        image_buffer[buffer_index, 3, :, :] = depthA
        image_buffer[buffer_index, 4:7, :, :] = rgbB
        image_buffer[buffer_index, 7, :, :] = depthB
        prior_buffer[buffer_index] = initial_pose.to_parameters(isQuaternion=True)
        label_buffer[buffer_index] = self.normalize_label(transformed_pose.to_parameters())

    def get_batch_qty(self):
        return math.ceil(self.size() / self.minibatch_size)

    def normalize_label(self, params):
        params[:3] /= float(self.metadata["translation_range"])
        params[3:] /= float(self.metadata["rotation_range"])
        return params

    """
        PARALLEL MINIBATCH METHODS
    """
    def compute_minibatches_permutations_(self):
        permutations = np.random.permutation(np.arange(0, self.size()))
        if self.max_size != 0:
            size = min(self.size(), self.max_size)
            permutations = permutations[:size]
        return [permutations[x:x + self.minibatch_size] for x in range(0, len(permutations), self.minibatch_size)]

    def load_minibatch(self, task):
        try:
            image_buffer = np.ndarray((len(task), 8, int(self.metadata["image_size"]), int(self.metadata["image_size"])), dtype=np.float32)
            prior_buffer = np.ndarray((len(task), 7), dtype=np.float32)
            label_buffer = np.ndarray((len(task), 6), dtype=np.float32)
            for buffer_index, permutation in enumerate(task):
                self.get_sample(permutation, image_buffer, prior_buffer, label_buffer, buffer_index)
        except Exception as e:
            print("Thread error : {}".format(e))
        return image_buffer, prior_buffer, label_buffer
