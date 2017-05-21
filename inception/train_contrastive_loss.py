#!/usr/local/bin/python3
"""
Training program.
"""
import datetime
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3

from data_shapenet_contrastive import Data
from contrastive_loss import contrastive_loss
from summarizer import ScalarSummarizer

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "{}/data".format(DEFAULT_DIR),
                           "Data directory")
tf.app.flags.DEFINE_string("chkpnt_dir", "{}/chkpnts".format(DEFAULT_DIR),
                           "Checkpoint directory")
tf.app.flags.DEFINE_string("summ_dir", "{}/summary".format(DEFAULT_DIR),
                           "Summary directory")
tf.app.flags.DEFINE_string("ranklist_dir", "{}/ranklist".format(DEFAULT_DIR),
                           "Ranklist directory")
tf.app.flags.DEFINE_string("no_classes", 100, "Number of classes")
tf.app.flags.DEFINE_string("margin", 0.1, "Margin size")
tf.app.flags.DEFINE_integer("from_step", -1,
                            "Continue training from a checkpoint")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size")
tf.app.flags.DEFINE_float("lr", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("lr_decay", 0.1, "Learning rate decay factor")
tf.app.flags.DEFINE_float("no_epochs_decay", 50, "Number of epochs per decay")
tf.app.flags.DEFINE_float("beta", 0.004, "Beta")
tf.app.flags.DEFINE_integer("log_period", 5, "Log period")
tf.app.flags.DEFINE_integer("val_period", 25, "Validation period")
tf.app.flags.DEFINE_integer("save_period", 1000, "Saving period")
tf.app.flags.DEFINE_integer("no_epochs", 20000, "Number of epoches")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")


def _log(message):
    if FLAGS.verbose:
        print(message)


def _get_feature_op(inputs):
    logits, _ = inception_v3(inputs, num_classes=FLAGS.no_classes)
    return logits


def _get_loss_op(left_logits, right_logits, similarities):
    loss_fn = contrastive_loss(FLAGS.margin)
    return loss_fn(left_logits, right_logits, similarities.astype(float))


def _get_train_op(loss_op, learning_rate, global_step):
    return tf.train.AdamOptimizer(learning_rate).minimize(
        loss_op, global_step=global_step)


def _train(
        train_dat,
        valid_dat,
        test_dat):

    train_size = train_dat.size()
    _log("training size = {}".format(train_size))
    _log("validation size = {}".format(valid_dat.size()))

    # Create a new graph
    graph = tf.Graph()
    with graph.as_default():
        # Update step
        step = FLAGS.from_step + 1

        _log("initializing the model")

        # ???
        decay_steps = train_size / FLAGS.batch_size * FLAGS.no_epochs_decay

        # Place holders
        left_inputs = tf.placeholder("float32", shape=(None, 299, 299, 3))
        right_inputs = tf.placeholder("float32", shape=(None, 299, 299, 3))
        left_labels = tf.placeholder("float32", shape=(None, 100))
        right_labels = tf.placeholder("float32", shape=(None, 100))

        # Tensors
        global_step = tf.Variable(step, trainable=False)
        learning_rate = tf.train.exponential_decay(
            FLAGS.lr,
            global_step,
            decay_steps,
            FLAGS.lr_decay)

        left_features = _get_feature_op(left_inputs)
        right_features = _get_feature_op(right_inputs)
        similarities = tf.equal(
            tf.argmax(
                left_labels, 1), tf.argmax(
                right_labels, 1))

        loss_op = _get_loss_op(
            left_features,
            right_features,
            similarities)
        train_op = _get_train_op(loss_op, learning_rate, global_step)

        # Start a session
        sess = tf.Session(graph=graph)

        summ = ScalarSummarizer(FLAGS.summ_dir, sess, {
            "training_loss": "float32",
            "validation_loss": "float32"
        })

        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())
        if FLAGS.from_step >= 0:
            _log("restoring the model")
            saver.restore(sess, os.path.join(
                FLAGS.chkpnt_dir, "inception-contrastive-{}".format(str(FLAGS.from_step))))

        for epoch in range(FLAGS.no_epochs):
            _log("{} epoch = {}".format(datetime.datetime.now(), epoch))
            train_dat.shuffle()

            for tr_left_inputs, tr_left_categories, tr_right_inputs, tr_right_catetories in train_dat.batches(
                    FLAGS.batch_size):
                # Create food
                food = {
                    left_inputs: tr_left_inputs,
                    right_inputs: tr_right_inputs,
                    left_labels: tr_left_categories,
                    right_labels: tr_right_catetories
                }

                # Feed the model
                loss_val = sess.run(loss_op,
                                    feed_dict=food)

                summ.log({"training_loss": loss_val}, step)

                # Log info
                if step % FLAGS.log_period == 0:
                    _log("-TRAIN- {} step={}, loss={:.4}"
                         .format(datetime.datetime.now(), step, loss_val))

                # Save the current model
                if step > 0 and step % FLAGS.save_period == 0:
                    _log("-SAVE- start")
                    saver.save(
                        sess,
                        os.path.join(
                            FLAGS.chkpnt_dir,
                            "inception-contrastive-"),
                        global_step=step)
                    _log("-SAVE- done")

                # Perform validation
                if step > 0 and step % FLAGS.val_period == 0:
                    _log("-VALID- {} start".format(datetime.datetime.now()))
                    valid_dat.shuffle()
                    losses = []
                    for vl_left_inputs, vl_left_categories, vl_right_inputs, vl_right_categories in valid_dat.batches(
                            FLAGS.batch_size):
                        # Create food
                        food = {
                            left_inputs: vl_left_inputs,
                            right_inputs: vl_right_inputs,
                            left_labels: vl_left_categories,
                            right_labels: vl_right_catetories
                        }

                        loss_val, = sess.run([loss_op], feed_dict=food)
                        losses.append(loss_val)

                    val_loss = np.mean(np.array(losses))

                    _log("-VALID- {} done: loss={:.4}"
                         .format(datetime.datetime.now(), val_loss))

                    summ.log({"validation_loss": val_loss}, step)

                step += 1

        # Save at the last step
        saver.save(
            sess,
            os.path.join(
                FLAGS.chkpnt_dir,
                "inception-contrastive-"),
            global_step=step)


def main():
    train_dat = Data(FLAGS.data_dir, "train",
                     no_categories=FLAGS.no_classes, suffix=".jpg")
    valid_dat = Data(FLAGS.data_dir, "valid",
                     no_categories=FLAGS.no_classes, suffix=".jpg")
    test_dat = Data(FLAGS.data_dir, "test", is_test=True,
                    no_categories=FLAGS.no_classes, suffix=".jpg")
    _train(
        train_dat,
        valid_dat,
        test_dat)


if __name__ == "__main__":
    main()
