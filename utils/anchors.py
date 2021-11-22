from enum import Enum

class Anchors(Enum):
    YOLOV3 = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    YOLOV4 = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    YOLOV3TINY = [10, 14,  23, 27,  37, 58, 81, 82,  135, 169, 344, 319]
    # Anchors for YOLOV4TINY are deliberately wrong, in order to match the wrong masks in CFG for pretrained YoloV4-tiny
    # from AlexeyAB repository, and keep the functionality of the previous 2 repositories from TNTWEN and mystic123.
    # In this repository, we provide a flag to override this anchors!
    YOLOV4TINY = [23, 27,  37, 58, 81, 82, 81, 82, 135, 169, 344, 319]