import PyTorchHelpers

from deeptracking.data.dataaugmentation import DataAugmentation
from deeptracking.data.dataset_utils import show_frames_from_buffer
from deeptracking.utils.argumentparser import ArgumentParser
from deeptracking.data.dataset import Dataset
import sys
import json
import logging
import logging.config
import numpy as np
import math
from datetime import datetime
import os
import time

from deeptracking.utils.data_logger import DataLogger
from deeptracking.utils.slack_logger import SlackLogger


def get_current_time(with_dashes=False):
    string = '%Y/%m/%d %H:%M:%S'
    if with_dashes:
        string = '%Y-%m-%d_%H-%M-%S'
    return datetime.now().strftime(string)


def config_logging(data):
    logging_filename = "{}.log".format(get_current_time(with_dashes=True))
    logging_path = data["logging"]["path"]
    path = os.path.join(logging_path, logging_filename)
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
    dictLogConfig = {
        "version": 1,
        'disable_existing_loggers': False,
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "basic_formatter",
                "stream": 'ext://sys.stdout',
            },
            "fileHandler": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": path,
            },
        },
        "loggers": {
            __name__: {
                "handlers": ["fileHandler", "default"],
                "level": data["logging"]["level"],
                "propagate": False
            }
        },

        "formatters": {
            "basic_formatter": {
                'format': '[%(levelname)s] %(message)s',
            },
            "detailed": {
                'format': '%(asctime)s %(name)s[%(levelname)s] %(filename)s:%(lineno)d %(message)s',
                'datefmt': "%Y-%m-%d %H:%M:%S",
            }
        }
    }
    logging.setLoggerClass(SlackLogger)
    logger = logging.getLogger(__name__)
    logging.config.dictConfig(dictLogConfig)
    return logger


def config_datasets(data):
    train_path = data["train_path"]
    valid_path = data["valid_path"]
    minibatch_size = int(data["minibatch_size"])
    rgb_noise = float(data["data_augmentation"]["rgb_noise"])
    depth_noise = float(data["data_augmentation"]["depth_noise"])
    occluder_path = data["data_augmentation"]["occluder_path"]
    background_path = data["data_augmentation"]["background_path"]
    blur_noise = int(data["data_augmentation"]["blur_noise"])
    h_noise = float(data["data_augmentation"]["h_noise"])
    s_noise = float(data["data_augmentation"]["s_noise"])
    v_noise = float(data["data_augmentation"]["v_noise"])
    channel_hide = data["data_augmentation"]["channel_hide"] == "True"

    data_augmentation = DataAugmentation()
    data_augmentation.set_rgb_noise(rgb_noise)
    data_augmentation.set_depth_noise(depth_noise)
    if occluder_path != "":
        data_augmentation.set_occluder(occluder_path)
    if background_path != "":
        data_augmentation.set_background(background_path)
    if channel_hide:
        data_augmentation.set_channel_hide(0.25)
    data_augmentation.set_blur(blur_noise)
    data_augmentation.set_hsv_noise(h_noise, s_noise, v_noise)

    message_logger.info("Setup Train : {}".format(train_path))
    train_dataset = Dataset(train_path, minibatch_size=minibatch_size)
    if not train_dataset.load():
        message_logger.error("Train dataset empty")
        sys.exit(-1)
    train_dataset.set_data_augmentation(data_augmentation)
    train_dataset.compute_mean_std()
    message_logger.info("Computed mean : {}\nComputed Std : {}".format(train_dataset.mean, train_dataset.std))
    message_logger.info("Setup Valid : {}".format(valid_path))
    valid_dataset = Dataset(valid_path, minibatch_size=minibatch_size, max_samples=20000)
    if not valid_dataset.load():
        message_logger.error("Valid dataset empty")
        sys.exit(-1)
    valid_dataset.set_data_augmentation(data_augmentation)
    valid_dataset.mean = train_dataset.mean
    valid_dataset.std = train_dataset.std
    return train_dataset, valid_dataset


def config_model(data, dataset):
    dataset_metadata = dataset.metadata
    gpu_device = int(data["gpu_device"])
    learning_rate = float(data["training_param"]["learning_rate"])
    learning_rate_decay = float(data["training_param"]["learning_rate_decay"])
    weight_decay = float(data["training_param"]["weight_decay"])
    input_size = int(data["training_param"]["input_size"])
    linear_size = int(data["training_param"]["linear_size"])
    convo1_size = int(data["training_param"]["convo1_size"])
    convo2_size = int(data["training_param"]["convo2_size"])
    model_finetune = data["model_finetune"]
    model_class = PyTorchHelpers.load_lua_class(data["training_param"]["file"], 'RGBDTracker')
    tracker_model = model_class('cuda', 'adam', gpu_device)

    tracker_model.set_configs({
        "input_size": input_size,
        "linear_size": linear_size,
        "convo1_size": convo1_size,
        "convo2_size": convo2_size,
    })

    if model_finetune == "":
        tracker_model.build_model()
        tracker_model.init_model()
    else:
        tracker_model.load(model_finetune)

    tracker_model.set_configs({
        "learningRate": learning_rate,
        "learningRateDecay": learning_rate_decay,
        "weightDecay": weight_decay,
        # Necessary data at test time, the user can get all information while loading the model and its configs
        "translation_range": float(dataset_metadata["translation_range"]),
        "rotation_range": float(dataset_metadata["rotation_range"]),
        "render_scale": dataset_metadata["object_width"],
        "mean_matrix": dataset.mean,
        "std_matrix": dataset.std
    })
    return tracker_model


def train_loop(model, dataset, logger, log_message_ratio=0.01):
    with dataset:
        batch_message_intervals = math.ceil(float(dataset.get_batch_qty()) * log_message_ratio)
        minibatchs = dataset.get_minibatch()
        start_time = time.time()
        for i, minibatch in enumerate(minibatchs):
            image_buffer, prior_buffer, label_buffer = minibatch
            if args.verbose:
                print("Train")
                print("Prior : {}".format(prior_buffer[0]))
                print("Label : {}".format(label_buffer[0]))
                show_frames_from_buffer(image_buffer, dataset.mean, dataset.std)

            losses = model.train([image_buffer, prior_buffer], label_buffer)
            statistics = model.extract_grad_statistic()
            logger.add_row("Minibatch", [losses["label"]])
            logger.add_row_from_dict("Grad_Rotation", statistics[1])
            logger.add_row_from_dict("Grad_Translation", statistics[2])
            if i % batch_message_intervals == 0:
                progression = float(i+1)/float(dataset.get_batch_qty())*100
                message_logger.info("[{}%] : Train loss: {}".format(int(progression), losses["label"]))
                elapsed_time = time.time() - start_time
                message_logger.info("Time/batch : {}h".format((100 * elapsed_time / progression)/3600))

    total_loss = data_logger.get_as_numpy("Minibatch")[:, 0]
    mean_loss = 0 if len(total_loss) < 5 else np.mean(total_loss[-5:])
    return mean_loss


def validation_loop(model, dataset):
    with dataset:
        loss_sum = 0
        loss_qty = 0
        minibatchs = dataset.get_minibatch()
        for image_buffer, prior_buffer, label_buffer in minibatchs:
            if args.verbose:
                print("Valid")
                print("Prior : {}".format(prior_buffer[0]))
                print("Label : {}".format(label_buffer[0]))
                show_frames_from_buffer(image_buffer, dataset.mean, dataset.std)
            prediction = model.test([image_buffer, prior_buffer])
            losses = model.loss_function(prediction, label_buffer)
            loss_sum += losses["label"]
            loss_qty += 1
    return loss_sum / loss_qty

if __name__ == '__main__':
    args = ArgumentParser(sys.argv[1:])
    if args.help:
        args.print_help()
        sys.exit(1)

    with open(args.config_file) as data_file:
        data = json.load(data_file)

    message_logger = config_logging(data)
    data_logger = DataLogger()
    data_logger.create_dataframe("Epoch", ["Train", "Valid"])
    data_logger.create_dataframe("Minibatch", ["Train"])
    data_logger.create_dataframe("Grad_Rotation", ["grad_rot_mean", "grad_rot_median", "grad_rot_min", "grad_rot_max"])
    data_logger.create_dataframe("Grad_Translation", ["grad_trans_mean", "grad_trans_median", "grad_trans_min", "grad_trans_max"])

    message_logger.info("Setup Datasets")
    train_dataset, valid_dataset = config_datasets(data)

    message_logger.info("Setup Model")
    tracker_model = config_model(data, train_dataset)
    message_logger.debug(tracker_model.model_string())

    MAX_EPOCH = int(data["max_epoch"])
    OUTPUT_PATH = data["output_path"]
    EARLY_STOP_WAIT_LIMIT = int(data["early_stop_wait_limit"])

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    message_logger.slack("Train start at {}".format(get_current_time()))
    best_validation_loss = 1000
    best_epoch = 0
    early_stop_wait = 0
    for epoch in range(MAX_EPOCH):
        train_loss = train_loop(tracker_model, train_dataset, data_logger)
        val_loss = validation_loop(tracker_model, valid_dataset)
        message_logger.slack("[Epoch {}] Train loss: {} Val loss: {}".format(epoch, train_loss, val_loss))
        data_logger.add_row("Epoch", [train_loss, val_loss])
        data_logger.save(OUTPUT_PATH)
        # Early Stop
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch
            tracker_model.save(os.path.join(OUTPUT_PATH, data["session_name"]), str(epoch))
            early_stop_wait = 0
        else:
            early_stop_wait += 1
            if early_stop_wait > EARLY_STOP_WAIT_LIMIT:
                break
    message_logger.slack("Train Terminated at {}".format(get_current_time()))
    message_logger.slack("Total Epoch: {}\nBest Validation Loss: {}".format(best_epoch, best_validation_loss))

