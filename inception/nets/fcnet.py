"""
An SVM class using fully-connected layers.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


class FCNet:
    def __init__(self, dims, beta):
        self._dims = dims
        self._beta = beta

    def forward(self, inputs):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        outputs = inputs
        for dim in self._dims:
            outputs = slim.layers.fully_connected(
                outputs,
                dim,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(
                    self._beta))
            outputs = slim.dropout(outputs, keep_prob)
        return outputs
