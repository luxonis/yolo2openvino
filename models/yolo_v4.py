import tensorflow as tf
from models.common import _conv2d_fixed_padding, _fixed_padding, _get_size, \
    _detection_layer, _upsample, _spp_block

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1


def _yolo_res_Block(inputs,in_channels,res_num,data_format,double_ch=False):
    out_channels = in_channels
    if double_ch:
        out_channels = in_channels * 2
    net = _conv2d_fixed_padding(inputs,in_channels*2,kernel_size=3,strides=2)
    route = _conv2d_fixed_padding(net,out_channels,kernel_size=1)
    net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)

    for _ in range(res_num):
        tmp=net
        net = _conv2d_fixed_padding(net,in_channels,kernel_size=1)
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=3)
        #shortcut
        net = tmp+net

    net=_conv2d_fixed_padding(net,out_channels,kernel_size=1)

    #concat
    net=tf.concat([net,route],axis=1 if data_format == 'NCHW' else 3)
    net=_conv2d_fixed_padding(net,in_channels*2,kernel_size=1)
    return net

def _yolo_conv_block(net,in_channels,a,b):
    for _ in range(a):
        out_channels=in_channels/2
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)
        net = _conv2d_fixed_padding(net,in_channels,kernel_size=3)

    out_channels=in_channels
    for _ in range(b):
        out_channels=out_channels/2
        net = _conv2d_fixed_padding(net,out_channels,kernel_size=1)

    return net

def csp_darknet53(inputs,data_format,batch_norm_params):
    """
    Builds CSPDarknet-53 model.activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)
    """
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x:x* tf.math.tanh(tf.math.softplus(x))):
        net = _conv2d_fixed_padding(inputs,32,kernel_size=3)
        #downsample
        #res1
        net=_yolo_res_Block(net,32,1,data_format,double_ch=True)
        #res2
        net = _yolo_res_Block(net,64,2,data_format)
        #res8
        net = _yolo_res_Block(net,128,8,data_format)

        #features of 54 layer
        up_route_54=net
        #res8
        net = _yolo_res_Block(net,256,8,data_format)
        #featyres of 85 layer
        up_route_85=net
        #res4
        net=_yolo_res_Block(net,512,4,data_format)

    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        biases_initializer=None,
                        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
        ########
        net = _yolo_conv_block(net,1024,1,1)

        net=_spp_block(net,data_format=data_format)

        net=_conv2d_fixed_padding(net,512,kernel_size=1)
        net = _conv2d_fixed_padding(net, 1024, kernel_size=3)
        net = _conv2d_fixed_padding(net, 512, kernel_size=1)

        #features of 116 layer
        route_3=net

        net = _conv2d_fixed_padding(net,256,kernel_size=1)
        upsample_size = up_route_85.get_shape().as_list()
        net = _upsample(net, upsample_size, data_format)
        route= _conv2d_fixed_padding(up_route_85,256,kernel_size=1)

        net = tf.concat([route,net], axis=1 if data_format == 'NCHW' else 3)
        net = _yolo_conv_block(net,512,2,1)
        #features of 126 layer
        route_2=net

        net = _conv2d_fixed_padding(net,128,kernel_size=1)
        upsample_size = up_route_54.get_shape().as_list()
        net = _upsample(net, upsample_size, data_format)
        route= _conv2d_fixed_padding(up_route_54,128,kernel_size=1)
        net = tf.concat([route,net], axis=1 if data_format == 'NCHW' else 3)
        net = _yolo_conv_block(net,256,2,1)
        #features of 136 layer
        route_1 = net

    return route_1, route_2, route_3


def yolo_v4(inputs, num_classes, anchors, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v4 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param num_classes: anchors.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :param with_spp: whether or not is using spp layer.
    :return:
    """

    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], data_format=data_format, reuse=reuse):

            #weights_regularizer=slim.l2_regularizer(weight_decay)
            #weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
        with tf.variable_scope('cspdarknet-53'):
            route_1, route_2, route_3 = csp_darknet53(inputs,data_format,batch_norm_params)

        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('yolo-v4'):
                #features of y1
                net = _conv2d_fixed_padding(route_1,256,kernel_size=3)
                detect_1 = _detection_layer(
                    net, num_classes, anchors[0:3], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')

                #features of y2
                net = _conv2d_fixed_padding(route_1, 256, kernel_size=3,strides=2)
                net=tf.concat([net,route_2], axis=1 if data_format == 'NCHW' else 3)
                net=_yolo_conv_block(net,512,2,1)
                route_147 =net
                net = _conv2d_fixed_padding(net,512,kernel_size=3)
                detect_2 = _detection_layer(
                    net, num_classes, anchors[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')

                # features of  y3
                net=_conv2d_fixed_padding(route_147,512,strides=2,kernel_size=3)
                net = tf.concat([net, route_3], axis=1 if data_format == 'NCHW' else 3)
                net = _yolo_conv_block(net,1024,3,0)
                detect_3 = _detection_layer(
                    net, num_classes, anchors[6:9], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')

                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')
                return detections

