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
* YoloV4-tiny-custom - [`CFG`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-custom.cfg).
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
```python
python convert_weights_pb.py \
--yolo 3 \
--class_names coco.names \
--output yolov3.pb \
--weights_file yolov3.weights \
--size 416  
```
2. Conversion of YoloV4-tiny model with custom input shapes and anchors (model trained with [`yolov4-tiny-custom.cfg`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-custom.cfg)):
```python
python convert_weights_pb.py \
--yolo 4 \
--weights_file yolov4-tiny_best.weights \
--class_names obj.names \
--output yolov4tiny.pb \
--tiny \
-h 320 \
-w 512 \
-a 10,14,23,27,37,58,81,82,135,169,344,319
```

#### Anchors

To correctly set up the anchors please look into your CFG file used in training, and search for the last `[yolo]` layer. You can find setting for anchors and mask. Mask represent the indices of the anchors. For example, if anchors are `10,14, 23,27, 37,58, 81,82, 135,169, 344,319`, then the mask `0,1,2` correspond to `10,14, 23,27, 37,58`. If mask is set to `1,2,3` it corresponds to `23,27, 37,58, 81,82`.

### 2. Conversion to OpenVINO

The result of the first step will be a frozen model (file ending in *.pb*). To generate the OpenVINO IR, OpenVINO must be installed. This repository is tested using OpenVINO 2021.3 version. 

#### JSON configuration

Before conversion you must edit the JSON file to match the values in the CFG. Below you can see the example of a JSON for [`yolov4-tiny.cfg`](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg). Please make sure that `classes` match the number of your objects, `anchors` match the anchors in `[yolo]` layer of your CFG, `masks` match the mask setting in `[yolo]` layers of your CFG, and `entry_points` correspond to the correct Yolo version. If you are using a tiny model, there should be 2 `[yolo]` layers and 3 otherwise. Anchors should be the same in all layers, while mask are different for each `[yolo]` layer. Please ensure that masks are set correctly in the JSON and in order that they appear.

```json
[
  {
    "id": "TFYOLOV3",
    "match_kind": "general",
    "custom_attributes": {
      "classes": 80,
      "anchors": [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319],
      "coords": 4,
      "num": 6,
      "masks": [[3, 4, 5], [1, 2, 3]],
      "entry_points": ["detector/yolo-v4-tiny/Reshape", "detector/yolo-v4-tiny/Reshape_4"]
    }
  }
]
```

If you have not changed anchors and masks, we already provide JSONs ready for conversion. You only have to change the `classes` attribute and specify path to the correct JSON during the conversion.

#### Conversion

Once you obtain a frozen model, you can convert it to OpenVINO IR by executing:

```python
python path_to_openvino/deployment_tools/model_optimizer/mo.py \
--input_model yolov3.pb \
--tensorflow_use_custom_operations_config json/yolo_v3.json \
--batch 1 \
--data_type FP16 \
--reverse_input_channel \
--model_name yolov3 \
--output_dir output_dir
```

where:

* `path_to_openvino`: path to your installation of OpenVINO,
* `input model`: path to your frozen model generated in the first step,
* `output_dir`: path where OpenVINO IR (xml and bin) will be saved,
* `tensorflow_use_custom_operations_config`: path to the JSON file with values that match CFG used in training.

### 3. Conversion to blob for inference on OAK devices

Once conversion to OpenVINO IR is successful, you can convert the OpenVINO IR to the *.blob* that can run on OAK devices. 

#### Web app

You can convert the model using our [blobconverter](https://blobconverter.luxonis.com/) web app. Select *OpenVINO 2021.3* > *OpenVINO Model* > *Continue*, upload *.xml* and *.bin*, and convert.

#### Python API

For this step we use the `blobconverter` library, which you can install using `pip install blobconverter`. Then you can convert the OpenVINO to IR using:

```python
blob_path = blobconverter.from_openvino(
    xml=path_to_xml,
    bin=path_to_bin,
    data_type="FP16",
    shaves=6,
    version="2021.3",
    use_cache=False
)
```

where `path_to_xml` and `path_to_bin` are paths to *.xml* and *.bin*, respectively.

## Debugging

If you encounter an issue with the conversion, please ensure you are correctly setting the flags as described in the above steps. If conversion still fails, search for relevant issues in:

* [tensorflow-yolo-v3/issues](https://github.com/mystic123/tensorflow-yolo-v3/issues) for issues with YoloV3 (tiny), 
* and [https://github.com/AlexeyAB/darknet/issues](AlexeyAB/darknet/issues) for issues with YoloV4 (tiny).

If you cannot find a solution for your problem, please open a new issue here on GitHub.

## Environment

Tested on:

```
Python 3.7
Tensorflow 1.14.0
Numpy 1.16.6
OpenVINO 2021.3
Pillow
```

