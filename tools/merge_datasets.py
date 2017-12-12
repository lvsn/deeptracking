from deeptracking.data.dataset import Dataset
from tqdm import tqdm
import os


if __name__ == '__main__':
    datasets_path = ["/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking_train/dragon/real_large/train",
                     "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking_train/dragon/synth_large/train"]
    output_path =    "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking_train/dragon/real_synth_large/train"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("Merging datasets :")
    for path in datasets_path:
        print(path)
    print("Into : {}".format(output_path))

    # load datasets
    datasets = [Dataset(x) for x in datasets_path]
    for dataset in datasets:
        dataset.load()

    # metadata sanity check
    for dataset_check in datasets:
        for other_dataset in datasets:
            if dataset_check.metadata != other_dataset.metadata:
                print(dataset_check.metadata)
                raise RuntimeError("Dataset {} have different metadata than {}\n{}\n{}".format(dataset_check.path,
                                                                                               other_dataset.path,
                                                                                               dataset_check.metadata,
                                                                                               other_dataset.metadata))

    metadata = datasets[0].metadata
    camera = datasets[0].camera
    output_dataset = Dataset(output_path, frame_class=metadata["save_type"])
    output_dataset.camera = camera
    output_dataset.metadata = metadata

    # transfer data
    for dataset in datasets:
        print("Process dataset {}".format(dataset.path))
        for i in tqdm(range(dataset.size())):
            rgbA, depthA, initial_pose = dataset.load_image(i)
            rgbB, depthB, transformed_pose = dataset.load_pair(i, 0)

            index = output_dataset.add_pose(rgbA, depthA, initial_pose)
            output_dataset.add_pair(rgbB, depthB, transformed_pose, index)

            if i % 500 == 0:
                output_dataset.dump_images_on_disk()
            if i % 5000 == 0:
                output_dataset.save_json_files(metadata)

    output_dataset.dump_images_on_disk()
    output_dataset.save_json_files(metadata)