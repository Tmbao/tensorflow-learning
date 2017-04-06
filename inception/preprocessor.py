#!/usr/local/bin/python3
"""
Training program.
"""
import os

import numpy as np
import tensorflow as tf

from data import Data
from nets.fc_aggregator import FCAggregator
from nets.inception_v3 import InceptionV3
from summarizer import ScalarSummarizer

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "{}/data".format(DEFAULT_DIR),
                           "Data directory")
tf.app.flags.DEFINE_integer("no_views", 26, "Number of views")


def _log(message):
    print(message)


def _train(
        train_dat,
        valid_dat,
        no_views):

    train_size = train_dat.size()
    _log("training size = {}".format(train_size))

    # Tensors
    cnn = InceptionV3()

    for b_inputs, _, b_paths in train_dat.batches(1):
        # Merge outputs from CNN
        concated_outputs = np.concatenate([np.expand_dims(cnn.classify(b_input), 0)
                                           for b_input in b_inputs], 0)
        view_pooling = np.amax(concated_outputs, 0)

        for index, path in enumerate(b_paths):
            np.save(os.path.join(path, "data"), view_pooling[index])
            _log("saved {}".format(os.path.join(path, "data")))

    for b_inputs, _, b_paths in valid_dat.batches(1):
        # Merge outputs from CNN
        concated_outputs = np.concatenate([np.expand_dims(cnn.classify(b_input), 0)
                                           for b_input in b_inputs], 0)
        view_pooling = np.amax(concated_outputs, 0)

        for index, path in enumerate(b_paths):
            np.save(os.path.join(path, "data"), view_pooling[index])
            _log("saved {}".format(os.path.join(path, "data")))


def main():
    train_dat = Data(FLAGS.data_dir, "train", FLAGS.no_views,
                     no_categories=16, suffix=".jpg")
    valid_dat = Data(FLAGS.data_dir, "valid", FLAGS.no_views,
                     no_categories=16, suffix=".jpg")
    _train(
        train_dat,
        valid_dat,
        FLAGS.no_views)


if __name__ == "__main__":
    main()
