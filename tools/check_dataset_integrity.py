"""
    Sanity tests for dataset folder
    - Make sure all images in viewpoints.json are in the folder
    ...
"""

from deeptracking.data.dataset import Dataset
import sys


if __name__ == '__main__':
    dataset_path = "/home/mathieu/Dataset/DeepTrack/skull"

    dataset = Dataset(dataset_path)
    if not dataset.load():
        print("[Error]: Train dataset empty")
        sys.exit(-1)

    # check if all viewpoints are there
    for frame, pose in dataset.data_pose:
        if not frame.exists(dataset.path):
            print("[Error]: Missing pose frame {}".format(frame.id))
            sys.exit(-1)

    # check if all pairs are there
    for key, value in dataset.data_pair.items():
        for frame, pose in value:
            if not frame.exists(dataset.path):
                print("[Error]: Missing pair frame {}".format(frame.id))
