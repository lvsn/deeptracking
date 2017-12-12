import PyTorchHelpers
import os
ESCAPE_KEY = 1048603
UNITY_DEMO = False


if __name__ == '__main__':
    class_path = "deeptracking/tracker/rgbd_tracker.lua"
    model_class = PyTorchHelpers.load_lua_class(class_path, 'RGBDTracker')
    tracker_model = model_class('cuda')

    model_path = "/home/mathieu/Dataset/DeepTrack/model/mixed_skull"
    input_model = "mixed_skull5"
    output_model = "mixed_skull_cpu"
    tracker_model.load(os.path.join(model_path, input_model))
    tracker_model.convert_backend("cpu")
    tracker_model.save(os.path.join(model_path, output_model))