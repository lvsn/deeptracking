# deeptracking

Torch + python implementation of [Deep 6-DOF Tracking](https://arxiv.org/abs/1703.09771)


The code to parse a ply file is not provided, this framework needs an implementation of [plyparser.py](https://github.com/lvsn/deeptracking/blob/develop/deeptracking/utils/plyparser.py)


## Useful links
- [Paper](https://arxiv.org/abs/1703.09771)
- [Dataset and project page](http://vision.gel.ulaval.ca/~jflalonde/projects/deepTracking/index.html)
- [Python code to download/load the data](https://github.com/lvsn/deeptrack_dataset)

## Generating data
### Generating synthetic data
Run:
```bash
python generate_synthetic_data.py config_file.json
```

#### dependencies
- cv2
- numpy (tested with 1.13.3)
- tqdm
- pyOpenGL
- glfw
- numpngw

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/generate_synthetic_example.json)

### Generating real data
Run:
```bash
python generate_real_data.py config_file.json
```

#### dependencies
- cv2
- numpy (tested with 1.13.3)
- tqdm
- pyOpenGL
- glfw
- numpngw

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/generate_real_example.json)

## Train
Run:
```bash
python train.py config_file.json
```

#### dependencies
- Hugh Perkins's [pytorch](https://github.com/hughperkins/pytorch)
- scipy, skimage, numpy (tested with 1.13.3)
- tqdm
- numpngw
- slackclient (could be removed)

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/train_example.json)

## Test
#### Sensor
Will run the tracker with a sensor (kinect 2)
```bash
python test_sensor.py config_file.json
```

#### Sequence
Will run the tracker on a sequence (folder) and save the error in a file
```bash
python test_sequence.py config_file.json
```

#### dependencies
- cv2
- numpy (tested with 1.13.3)
- Hugh Perkins's [pytorch](https://github.com/hughperkins/pytorch)
- pyOpenGL
- glfw
- numpngw
- [pyfreenect2](https://github.com/MathGaron/py3freenect2) (for Kinect2 sensor)

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/test_example.json)
