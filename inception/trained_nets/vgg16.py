"""
The Vgg15 model for TensorFlow.

This is a pre-trained Deep Neural Network for classifying images.
"""
import os

import numpy as np
import tensorflow as tf

_VGG_MEAN = [103.939, 116.779, 123.68]

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_NET_DIR = os.path.join(_SOURCE_DIR, "vgg16_net")
_GRAPH_DEF_FILE = "vgg16.npy"


class Vgg16:
    def __init__(self, net_dir=_NET_DIR, graph_def_file=_GRAPH_DEF_FILE):
        """
        Initialize the pretrained model.
        """
        self._data_dict = np.load(os.path.join(net_dir, graph_def_file), encoding='latin1').item()
        self._build()


    def classify(self, inputs):

        food = {self._rgb: inputs}
        return self._session.run(self._prob, feed_dict=food)
        

    def _build(self):

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._rgb = tf.placeholder(tf.float32, shape=(None, None, None, 3))
            
            rgb_scaled = tf.image.resize_images(self._rgb, tf.constant([224, 224], dtype=tf.int32))

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            bgr = tf.concat(axis=3, values=[
                blue - _VGG_MEAN[0],
                green - _VGG_MEAN[1],
                red - _VGG_MEAN[2],
            ])

            self._conv1_1 = self._conv_layer(bgr, "conv1_1")
            self._conv1_2 = self._conv_layer(self._conv1_1, "conv1_2")
            self._pool1 = self._max_pool(self._conv1_2, 'pool1')

            self._conv2_1 = self._conv_layer(self._pool1, "conv2_1")
            self._conv2_2 = self._conv_layer(self._conv2_1, "conv2_2")
            self._pool2 = self._max_pool(self._conv2_2, 'pool2')

            self._conv3_1 = self._conv_layer(self._pool2, "conv3_1")
            self._conv3_2 = self._conv_layer(self._conv3_1, "conv3_2")
            self._conv3_3 = self._conv_layer(self._conv3_2, "conv3_3")
            self._pool3 = self._max_pool(self._conv3_3, 'pool3')

            self._conv4_1 = self._conv_layer(self._pool3, "conv4_1")
            self._conv4_2 = self._conv_layer(self._conv4_1, "conv4_2")
            self._conv4_3 = self._conv_layer(self._conv4_2, "conv4_3")
            self._pool4 = self._max_pool(self._conv4_3, 'pool4')

            self._conv5_1 = self._conv_layer(self._pool4, "conv5_1")
            self._conv5_2 = self._conv_layer(self._conv5_1, "conv5_2")
            self._conv5_3 = self._conv_layer(self._conv5_2, "conv5_3")
            self._pool5 = self._max_pool(self._conv5_3, 'pool5')

            self._fc6 = self._fc_layer(self._pool5, "fc6")
            self._relu6 = tf.nn.relu(self._fc6)

            self._fc7 = self._fc_layer(self._relu6, "fc7")
            self._relu7 = tf.nn.relu(self._fc7)

            self._fc8 = self._fc_layer(self._relu7, "fc8")

            self._prob = tf.nn.softmax(self._fc8, name="prob")

        self._data_dict = None
        self._session = tf.Session(graph=self._graph)

    def _avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self._get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self._get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self._get_fc_weight(name)
            biases = self._get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def _get_conv_filter(self, name):
        return tf.constant(self._data_dict[name][0], name="filter")

    def _get_bias(self, name):
        return tf.constant(self._data_dict[name][1], name="biases")

    def _get_fc_weight(self, name):
        return tf.constant(self._data_dict[name][0], name="weights")