"""
A Multiview CNN class.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.models.inception import inception


class MVCNNet:
    def __init__(self, beta, no_views, no_classes):
        self._beta = beta
        self._no_views = no_views
        self._no_classes = no_classes

    def forward(self, inputs):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            outputs = [
                inception.inference(
                    input,
                    1000,
                    for_training=True,
                    restore_logits=True) for input in inputs]
            concated_outputs = tf.concatenate([tf.expand_dims(output, 0)
                                               for output in outputs], 0)
            view_pooling = tf.amax(concated_outputs, 0)
            fc1 = slim.layers.fully_connected(
                view_pooling,
                2048,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(
                    self._beta))
            fc2 = slim.layers.fully_connected(
                fc1,
                2048,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(
                    self._beta))
            fc3 = slim.layers.fully_connected(
                fc1,
                self._no_classes,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(
                    self._beta))

            return fc3
