import tensorflow as tf
from models import yolo_v3, yolo_v3_tiny, yolo_v4, yolo_v4_tiny

from utils.utils import load_weights, load_names, detections_boxes, freeze_graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'yolo', 3, 'Yolo version. Possible options: 3 or 4.'
)
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names. Usually a file ending in .names.')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with Yolo\' weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output', 'frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model output path')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of Yolo')
# tf.app.flags.DEFINE_bool(
#    'spp', False, 'Use SPP version of Yolo')
tf.app.flags.DEFINE_integer(
    'size', None, 'Image size. If both, height and width, are not provided, a square input shape of size as set with'
                 ' this flag will be used')
tf.app.flags.DEFINE_integer(
    'height', None, 'Input image height. If height is set, width must also be set. The size flag will be ignored.',
    short_name='h'
)
tf.app.flags.DEFINE_integer(
    'width', None, 'Input image width. If width is set, height must also be set. The size flag will be ignored.',
    short_name='w'
)


def main(argv=None):

    if FLAGS.yolo == 3:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3
    elif FLAGS.yolo == 4:
        if FLAGS.tiny:
            model = yolo_v4_tiny.yolo_v4_tiny
        else:
            model = yolo_v4.yolo_v4
    else:
        raise ValueError(f"{FLAGS.yolo} is not supported Yolo version. Supported versions: 3, 4.")

    classes = load_names(FLAGS.class_names)

    # set input shape
    if FLAGS.height is not None and FLAGS.width is not None:
        inputs = tf.placeholder(tf.float32, [None, FLAGS.height, FLAGS.width, 3], "inputs")
        if FLAGS.size is not None:
            print("Width and height are set, size flag will be ignored!")
    elif FLAGS.size is not None:
        inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], "inputs")
    else:
        print("Neither size nor width and height flags are set. Please specify input shape!")

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output_graph)

if __name__ == '__main__':
    tf.app.run()
