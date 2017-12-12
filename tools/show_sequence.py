from deeptracking.data.modelrenderer import InitOpenGL, ModelRenderer
from deeptracking.data.dataset import Dataset
from deeptracking.data.dataset_utils import image_blend
import sys
import cv2

ESCAPE_KEY = 1048603

if __name__ == '__main__':

    # Populate important data from config file
    VIDEO_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking/sequences/skull"

    models = [{
      "name": "turtle",
      "model_path": "/home/mathieu/Dataset/3D_models/skull/skull.ply",
      "ambiant_occlusion_model": "/home/mathieu/Dataset/3D_models/skull/skull_ao.ply",
      "object_width": "250"
    }]

    MODELS_3D = models
    SHADER_PATH = "/home/mathieu/source/deeptracking/deeptracking/data/shaders"

    OBJECT_WIDTH = int(MODELS_3D[0]["object_width"])
    MODEL_3D_PATH = MODELS_3D[0]["model_path"]
    try:
        MODEL_3D_AO_PATH = MODELS_3D[0]["ambiant_occlusion_model"]
    except KeyError:
        MODEL_3D_AO_PATH = None
    frame_download_path = None

    video_data = Dataset(VIDEO_PATH)
    if not video_data.load():
        print("[ERROR] Error while loading video...")
        sys.exit(-1)
    frame_download_path = video_data.path
    # Makes the list a generator for compatibility with sensor's generator
    gen = lambda alist: [(yield i) for i in alist]
    frame_generator = gen(video_data.data_pose)
    camera = video_data.camera

    # Renderer
    window = InitOpenGL(camera.width, camera.height)
    vpRender = ModelRenderer(MODEL_3D_PATH, SHADER_PATH, camera, window, (camera.width, camera.height))
    vpRender.load_ambiant_occlusion_map(MODEL_3D_AO_PATH)

    for i, (current_frame, ground_truth_pose) in enumerate(frame_generator):
        # get actual frame
        current_rgb, current_depth = current_frame.get_rgb_depth(frame_download_path)
        screen = current_rgb

        rgb_render, depth_render = vpRender.render(ground_truth_pose.transpose())

        blend = image_blend(rgb_render, current_rgb)

        cv2.imshow("Debug", blend[:, :, ::-1])

        key = cv2.waitKey(30)
        key_chr = chr(key & 255)
        if key != -1:
            print("pressed key id : {}, char : [{}]".format(key, key_chr))
        if key == ESCAPE_KEY:
            break
        if key == 1048608:
            print("Frame : {}".format(i))

