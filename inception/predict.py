#!python3
"""Carry out prediction."""

import tensorflow as tf

import cv2
from nets.inception_v3 import InceptionV3

tf.app.flags.DEFINE_string("input", "", "Specify an input image.")

FLAGS = tf.app.flags.FLAGS


def _check_flags():
    if not FLAGS.input:
        raise ValueError("You must specify an input image.")


def main(_):
    _check_flags()

    image = cv2.imread(FLAGS.input)
    network = InceptionV3()
    print(network.classify([image]).shape)


if __name__ == '__main__':
    tf.app.run()
