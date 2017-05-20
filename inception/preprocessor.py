#!/usr/local/bin/python3
"""
Training program.
"""
import os

import numpy as np
import tensorflow as tf

from data_shapenet import Data
from trained_nets.inception_v3 import InceptionV3
from trained_nets.vgg16 import Vgg16
from summarizer import ScalarSummarizer

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "{}/data".format(DEFAULT_DIR),
                           "Data directory")
tf.app.flags.DEFINE_integer("no_views", 26, "Number of views")
tf.app.flags.DEFINE_string(
    "model",
    "inceptionv3",
    "Model name [inceptionv3, vgg16]")
tf.app.flags.DEFINE_string(
    "summarizer",
    "pool",
    "Summarization method [pool, concat]")
tf.app.flags.DEFINE_boolean(
    "overwrite",
    False,
    "Should the program overwrite existing files?")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")


def _log(message):
    if FLAGS.verbose:
        print(message)


def _get_model():
    if FLAGS.model == "inceptionv3":
        return InceptionV3()
    elif FLAGS.model == "vgg16":
        return Vgg16()


def _summarize(inputs):
    if FLAGS.summarizer == "pool":
        concated_outputs = np.concatenate([np.expand_dims(input, 0)
                                           for input in inputs], 0)
        view_pooling = np.amax(concated_outputs, 0)
        return view_pooling
    elif FLAGS.summarizer == "concat":
        return np.concatenate(inputs, axis=1)
    else:
        raise ValueError(
            "Summarizer {} is not supported".format(
                FLAGS.summarizer))


def _train(
        train_dat,
        valid_dat,
        test_dat,
        no_views):

    train_size = train_dat.size()
    _log("training size = {}".format(train_size))

    # Tensors
    cnn = _get_model()

    for b_inputs, _, b_paths in train_dat.batches(1):
        outputs = [cnn.classify(b_input) for b_input in b_inputs]
        # Merge outputs from CNN
        outputs = _summarize(outputs)

        for index, path in enumerate(b_paths):
            np.save(
                os.path.join(
                    path,
                    "data.{}.{}".format(
                        FLAGS.model,
                        FLAGS.summarizer)),
                outputs[index])
            _log(
                "saved {}".format(
                    os.path.join(
                        path,
                        "data.{}.{}".format(
                            FLAGS.model,
                            FLAGS.summarizer))))

    for b_inputs, _, b_paths in valid_dat.batches(1):
        outputs = [cnn.classify(b_input) for b_input in b_inputs]
        # Merge outputs from CNN
        outputs = _summarize(outputs)

        for index, path in enumerate(b_paths):
            np.save(
                os.path.join(
                    path,
                    "data.{}.{}".format(
                        FLAGS.model,
                        FLAGS.summarizer)),
                outputs[index])
            _log(
                "saved {}".format(
                    os.path.join(
                        path,
                        "data.{}.{}".format(
                            FLAGS.model,
                            FLAGS.summarizer))))

    for b_inputs, _, b_paths in test_dat.batches(1):
        outputs = [cnn.classify(b_input) for b_input in b_inputs]
        # Merge outputs from CNN
        outputs = _summarize(outputs)

        for index, path in enumerate(b_paths):
            np.save(
                os.path.join(
                    path,
                    "data.{}.{}".format(
                        FLAGS.model,
                        FLAGS.summarizer)),
                outputs[index])
            _log(
                "saved {}".format(
                    os.path.join(
                        path,
                        "data.{}.{}".format(
                            FLAGS.model,
                            FLAGS.summarizer))))


def _filter_fn(prefix):
    if not FLAGS.overwrite:
        return not os.path.isfile(
            os.path.join(
                prefix,
                "data.{}.{}.npy".format(
                    FLAGS.model,
                    FLAGS.summarizer)))
    else:
        return True


def main():
    train_dat = Data(FLAGS.data_dir, "train", FLAGS.no_views,
                     no_categories=100, suffix=".jpg", filter_fn=_filter_fn)
    valid_dat = Data(FLAGS.data_dir, "valid", FLAGS.no_views,
                     no_categories=100, suffix=".jpg", filter_fn=_filter_fn)
    test_dat = Data(FLAGS.data_dir, "test", FLAGS.no_views,
                    no_categories=100, suffix=".jpg", filter_fn=_filter_fn)
    _train(
        train_dat,
        valid_dat,
        test_dat,
        FLAGS.no_views)


if __name__ == "__main__":
    main()
