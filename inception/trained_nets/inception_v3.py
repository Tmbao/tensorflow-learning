"""
The Inception Model v3 for TensorFlow.

This is a pre-trained Deep Neural Network for classifying images.
"""
import os

import numpy as np
import tensorflow as tf

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_NET_DIR = os.path.join(_SOURCE_DIR, "inception_v3_net")
_GRAPH_DEF_FILE = "classify_image_graph_def.pb"

_INPUT_TENSOR_NAME = "DecodeJpeg:0"
_FEATURE_TENSOR_NAME = "pool_3:0"
_SOFT_MAX_TENSOR_NAME = "softmax:0"


class InceptionV3:
    """
    The Inception model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the Inception model
    will be loaded and can be used immediately without training.

    The Inception model can also be used for Transfer Learning.
    """

    def __init__(self, net_dir=_NET_DIR, graph_def_file=_GRAPH_DEF_FILE):
        """
        Initialize the pretrained model.
        """
        # Create a new TF computational graph
        self._graph = tf.Graph()

        with self._graph.as_default():
            # Load the trained model
            path = os.path.join(net_dir, graph_def_file)
            with tf.gfile.FastGFile(path, "rb") as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name="")

        self._input = self._graph.get_tensor_by_name(_INPUT_TENSOR_NAME)
        self._outputs = self._graph.get_tensor_by_name(_FEATURE_TENSOR_NAME)
        self._session = tf.Session(graph=self._graph)

    def classify(self, inputs):
        return np.concatenate([self._session.run(self._outputs, feed_dict={self._input: input})
                               for input in inputs])
