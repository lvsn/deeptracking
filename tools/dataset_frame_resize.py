"""
    Resize all frames of a dataset
"""
from deeptracking.data.dataset import Dataset
import sys
import cv2
import os

if __name__ == '__main__':
    folder = "/home/mathieu/Dataset/DeepTrack/dragon/"
    dataset_path = os.path.join(folder, "train_raw_real")
    new_dataset_path = os.path.join(folder, "train_raw_real_resized")
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)

    dataset = Dataset(dataset_path)
    if not dataset.load():
        print("[Error]: Train dataset empty")
        sys.exit(-1)

    new_dataset = Dataset(new_dataset_path)
    new_dataset.camera = dataset.camera.copy()
    new_dataset.camera.set_ratio(2)
    for i in range(dataset.size()):
        rgb, depth, pose = dataset.load_image(i)
        new_rgb = cv2.resize(rgb, (new_dataset.camera.width, new_dataset.camera.height))
        new_depth = cv2.resize(depth, (new_dataset.camera.width, new_dataset.camera.height))
        new_dataset.add_pose(new_rgb, new_depth, pose)
        if i % (1*dataset.size()/100) == 0:
            print("Progress : {}%".format(i*100/dataset.size()))
    new_dataset.set_save_type(dataset.metadata["save_type"])
    new_dataset.dump_images_on_disk()
    new_dataset.save_json_files(dataset.metadata)
