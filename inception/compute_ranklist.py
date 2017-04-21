#!/usr/local/bin/python3
"""
Training program.
"""
import datetime
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_shapenet import Data
from nets.fcnet import FCNet

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "{}/data".format(DEFAULT_DIR),
                           "Data directory")
tf.app.flags.DEFINE_string("chkpnt_dir", "{}/chkpnts".format(DEFAULT_DIR),
                           "Checkpoint directory")
tf.app.flags.DEFINE_string("ranklist_dir", "{}/ranklist".format(DEFAULT_DIR),
                           "Ranklist directory")
tf.app.flags.DEFINE_integer("from_step", -1,
                            "Continue training from a checkpoint")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")


def _log(message):
    if FLAGS.verbose:
        print(message)


def _get_infer_op(logits):
    return tf.argmax(logits, 1)


def _compute(
        train_dat,
        valid_dat):

    train_size = train_dat.size()
    _log("training size = {}".format(train_size))
    _log("validation size = {}".format(valid_dat.size()))

    # Create a new graph
    graph = tf.Graph()
    with graph.as_default():

        # Start a session
        sess = tf.Session(graph=graph)

        _log("initializing the model")

        # Place holders
        inputs = tf.placeholder("float32", shape=(None, 2048))
        labels = tf.placeholder("float32", shape=(None, 100))

        # Tensors
        forward_op = FCNet(dims=[2048, 2048, 100], graph=graph, beta=FLAGS.beta).forward(inputs)
        infer_op = _get_infer_op(forward_op)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        if FLAGS.from_step >= 0:
            _log("restoring the model")
            saver.restore(sess, os.path.join(FLAGS.chkpnt_dir, str(FLAGS.from_step)))

        for val_inputs, val_labels, val_paths in valid_dat.batches(FLAGS.batch_size):
            # Create food
            food = {labels: val_labels, inputs: np.squeeze(val_inputs), "keep_prob:0": 1}

            infer_val, = sess.run(infer_op, feed_dict=food)

            for idx in range(FLAGS.batch_size):
                ranklist_path = os.path.join(FLAGS.ranklist_dir, os.paht.basename(val_paths[idx]))
                with open(ranklist_path, "w") as frl:
                    frl.write("\n".join(train_dat.groups(infer_val[idx])))
                _log("Wrote to {}.".format(ranklist_path))


def main():
    train_dat = Data(FLAGS.data_dir, "train", 1,
                     no_categories=100, suffix=".inceptionv3.pool.npy")
    valid_dat = Data(FLAGS.data_dir, "valid", 1,
                     no_categories=100, suffix=".inceptionv3.pool.npy")
    _compute(
        train_dat,
        valid_dat)


if __name__ == "__main__":
    main()
