# Yolo2OpenVINO

[![GitHub](https://img.shields.io/github/license/luxonis/depthai-model-zoo?color=blue&style=flat-square&label=LICENSE)](https://github.com/luxonis/yolo2openvino/blob/main/LICENSE)  [![stars](https://img.shields.io/github/stars/luxonis?affiliations=OWNER&label=LUXONIS%20STARS&style=flat-square)](https://github.com/luxonis)  [![web-interface](https://img.shields.io/static/v1?label=MORE&message=TUTORIALS&color=orange&style=flat-square)](https://github.com/luxonis/depthai-ml-training/)

![Logo](https://user-images.githubusercontent.com/56075061/142620908-580e2e06-eeb8-4222-a55f-b428cf76bd3d.png)

This repository contains implementation of YoloV3 and YoloV4 object detectors in Tensorflow in order to export them to OpenVINO IR. Repository is based on code from [mystic123/tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3) ([`Apache License 2.0`](https://github.com/mystic123/tensorflow-yolo-v3/blob/master/LICENSE)) and [TNTWEN/OpenVINO-YOLOV4](https://github.com/TNTWEN/OpenVINO-YOLOV4) ([`MIT License`](https://github.com/TNTWEN/OpenVINO-YOLOV4/blob/master/LICENSE)).

We merge and reorganize the code from both repositories, to provide both implementations in one place. We provide the instructions for conversion to OpenVINO IR. We also address the following issues/shortcomings from the past repositories:

* non-square input shape,
* anchor issues.

Check out [`luxonis/depthai-ml-training`](https://github.com/luxonis/depthai-ml-training/) for **tutorial** on how to **train YoloV3-tiny or YoloV4-tiny**.

## Pretrained weights

Pretrained weights and corresponding configs for YoloV4 and YoloV3 are available in [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) repository and at [pjreddie.com](https://pjreddie.com/darknet/yolo/), respectively.

* YoloV3-tiny - [`weights`](https://pjreddie.com/media/files/yolov3-tiny.weights),  [`CFG`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg).
* YoloV4-tiny - [`weights`](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights),  [`CFG`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg).
* YoloV3 - [`weights`](https://pjreddie.com/media/files/yolov3.weights),  [`CFG`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg).
* YoloV4 - [`weights`](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights),  [`CFG`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg).

**IMPORTANT:** Note that the mask in the last Yolo layer is wrongly set in the official pretrained version of YoloV4-tiny (`1,2,3` instead of `0,1,2`). Masks in this implementation are set to `0,1,2` ,`3,4,5` ,(and `6,7,8` for non-tiny versions). That is why the default anchors for YoloV4-tiny are "wrongly" set in our implementation. If you train the YoloV4 with [`yolov4-tiny-custom.cfg`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-custom.cfg) please provide custom anchors using `-a` flag (more info in the next section).

## Usage

### 1. Conversion to TF

Conversion is similar to the conversion in the original repositories, though additional flags are supported for non-square input shapes and custom anchors.

Supported flags:

* `--class_names`: path to the class names file,
* `--weights_file`: path to the weights file,
* `--data_format`: NCHW (gpu only) or NHWC (default),
* `--tiny`: use tiny version of Yolo if set,
* `--output_graph`: output location of the frozen model,
* `--size`: image size; if both, height and width, are not provided, a square input shape of `--size` will be used,
* `-h`, `--height`: input image height; `--width` must also be set with this flag,
* `-w`, `--width`: input image width; `--height` must also be set with this flag,
* `-a`, `--anchors`: list of anchors; if not set default anchors available in [utils/anchors.py](https://github.com/luxonis/yolo2openvino/blob/main/utils/utils.py) will be used. Anchors must be integers provided in the correct order, separated by `,` without spaces. See the second example in [examples](#examples).

#### Examples

1. Conversion of a pretrained YoloV3 model with square input shape:
```
python convert_weights_pb.py \
--yolo 3 \
--class_names coco.names \
--output yolov3.pb \
--weights_file yolov3.weights \
--size 416  
```
2. Conversion of YoloV4-tiny model with custom input shapes and anchors (model trained with [`yolov4-tiny-custom.cfg`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-custom.cfg)):
```
python convert_weights_pb.py \
--yolo 4 \ 
--weights_file yolov4-tiny_best.weights \
--class_names obj.names \
--output yolov4tiny.pb \
--tiny \
-h 320 \
-w 512 \
-a 10,14,23,27,37,58,81,82,135,169,344,319 \
```

### 2. Conversion to OpenVINO



## Environment

Tested on:

```
Python 3.8
Tensorflow 1.14.0.
Pillow 8.3.1
Numpy 1.16.6
```

