# Yolo2OpenVINO

[![GitHub](https://img.shields.io/github/license/luxonis/depthai-model-zoo?color=blue&style=flat-square&label=LICENSE)](https://github.com/luxonis/yolo2openvino/blob/main/LICENSE)  [![stars](https://img.shields.io/github/stars/luxonis?affiliations=OWNER&label=LUXONIS%20STARS&style=flat-square)](https://github.com/luxonis)  [![web-interface](https://img.shields.io/static/v1?label=MORE&message=TUTORIALS&color=orange&style=flat-square)](https://github.com/luxonis/depthai-ml-training/)

![Logo](https://user-images.githubusercontent.com/56075061/142620908-580e2e06-eeb8-4222-a55f-b428cf76bd3d.png)

This repository contains implementation of YoloV3 and YoloV4 object detectors in Tensorflow. Repository is based on code from [mystic123/tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3) ([`Apache License 2.0`](https://github.com/mystic123/tensorflow-yolo-v3/blob/master/LICENSE)) and [TNTWEN/OpenVINO-YOLOV4](https://github.com/TNTWEN/OpenVINO-YOLOV4) ([`MIT License`](https://github.com/TNTWEN/OpenVINO-YOLOV4/blob/master/LICENSE)).

We merge and reorganize the code from both repositories, to provide both implementations in one place. We provide the instructions for conversion to OpenVINO IR. We also address the following issues/shortcomings from the past repositories:

* non-square input shape,
* anchor issues.

Check out [`luxonis/depthai-ml-training`](https://github.com/luxonis/depthai-ml-training/) for **tutorial** on how to **train YoloV3-tiny or YoloV4-tiny**.

## Usage - TODO (edit old):
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download and convert model weights:    
    1. Download binary file with desired weights: 
        1. Full weights: `wget https://pjreddie.com/media/files/yolov3.weights`
        1. Tiny weights: `wget https://pjreddie.com/media/files/yolov3-tiny.weights` 
        1. SPP weights: `wget https://pjreddie.com/media/files/yolov3-spp.weights` 
    2. Run `python ./convert_weights.py` and `python ./convert_weights_pb.py`        
3. Run `python ./demo.py --input_img <path-to-image> --output_img <name-of-output-image> --frozen_model <path-to-frozen-model>`


####Optional Flags
1. convert_weights:
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov3-tiny
    5. `--spp`
        1. Use yolov3-spp
    6. `--ckpt_file`
        1. Output checkpoint file
2. convert_weights_pb.py:
    1. `--class_names`
            1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file    
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov3-tiny
    5. `--spp`
        1. Use yolov3-spp
    6. `--output_graph`
        1. Location to write the output .pb graph to
3. demo.py
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--ckpt_file`
        1. Path to the checkpoint file
    5. `--frozen_model`
        1. Path to the frozen model
    6. `--conf_threshold`
        1. Desired confidence threshold
    7. `--iou_threshold`
        1. Desired iou threshold
    8. `--gpu_memory_fraction`
        1. Fraction of gpu memory to work with

## Environment

Tested on:

```
Python 3.8
Tensorflow 1.14.0.
Pillow 8.3.1
Numpy 1.19.5
```

