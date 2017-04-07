"""
An SVM class using fully-connected layers.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


class FCNet:
    def __init__(self, dims, graph, beta):
        self._dims = dims
        self._graph = graph
        self._beta = beta

    def forward(self, inputs):
        with self._graph.as_default():
            outputs = inputs
            for dim in self._dims:
                outputs = slim.layers.fully_connected(outputs, dim, 
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(self._beta))
            return outputs