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
from summarizer import ScalarSummarizer

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "{}/data".format(DEFAULT_DIR),
                           "Data directory")
tf.app.flags.DEFINE_string("chkpnt_dir", "{}/chkpnts".format(DEFAULT_DIR),
                           "Checkpoint directory")
tf.app.flags.DEFINE_string("summ_dir", "{}/summary".format(DEFAULT_DIR),
                           "Summary directory")
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


def _get_loss_op(logits, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(losses)


def _get_train_op(loss_op, learning_rate, global_step):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss_op, global_step=global_step)


def _get_infer_op(logits):
    return tf.argmax(logits, 1)


def _train(
        train_dat,
        valid_dat,
        summ):

    train_size = train_dat.size()
    _log("training size = {}".format(train_size))
    _log("validation size = {}".format(valid_dat.size()))

    # Create a new graph
    graph = tf.Graph()
    with graph.as_default():
        # Update step
        step = FLAGS.from_step + 1

        # Start a session
        sess = tf.Session(graph=graph)

        _log("initializing the model")

        # ???
        decay_steps = train_size / FLAGS.batch_size * FLAGS.no_epochs_decay

        # Place holders
        inputs = tf.placeholder("float32", shape=(None, 2048))
        labels = tf.placeholder("float32", shape=(None, 100))

        # Tensors
        global_step = tf.Variable(step, trainable=False)
        learning_rate = tf.train.exponential_decay(
            FLAGS.lr,
            global_step,
            decay_steps,
            FLAGS.lr_decay)
        forward_op = FCNet(dims=[2048, 2048, 100], graph=graph, beta=FLAGS.beta).forward(inputs)
        reg_op = tf.add_n(tf.losses.get_regularization_losses())
        loss_op = _get_loss_op(forward_op, labels) + reg_op
        train_op = _get_train_op(loss_op, learning_rate, global_step)
        infer_op = _get_infer_op(forward_op)

        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())
        if FLAGS.from_step >= 0:
            _log("restoring the model")
            saver.restore(sess, os.path.join(FLAGS.chkpnt_dir, str(FLAGS.from_step)))

        for epoch in range(FLAGS.no_epochs):
            _log("{} epoch = {}".format(datetime.datetime.now(), epoch))
            train_dat.shuffle()

            for trn_inputs, trn_labels, _ in train_dat.batches(FLAGS.batch_size):
                trn_inputs = np.squeeze(trn_inputs)

                # Create food
                food = {labels: trn_labels, inputs: trn_inputs, "keep_prob:0": 0.5}

                # Feed the model
                _, _, loss_val = sess.run([train_op, infer_op, loss_op],
                                          feed_dict=food)

                summ.log({"training_loss": loss_val}, step)

                # Log info
                if step % FLAGS.log_period == 0:
                    _log("-TRAIN- {} step={}, loss={:.4}"
                         .format(datetime.datetime.now(), step, loss_val))

                # Save the current model
                if step % FLAGS.save_period == 0:
                    _log("-SAVE- start")
                    saver.save(sess, os.path.join(FLAGS.chkpnt_dir, str(step)), global_step=step)
                    _log("-SAVE- done")

                # Perform validation
                if step > 0 and step % FLAGS.val_period == 0:
                    _log("-VALID- {} start".format(datetime.datetime.now()))
                    valid_dat.shuffle()
                    ground_truth = []
                    predictions = []
                    losses = []
                    for val_inputs, val_labels, _ in valid_dat.batches(FLAGS.batch_size):

                        # Create food
                        food = {labels: val_labels, inputs: np.squeeze(val_inputs), "keep_prob:0": 1}

                        infer_val, loss_val, = sess.run(
                            [infer_op, loss_op], feed_dict=food)

                        ground_truth = np.concatenate(
                            [ground_truth, np.argmax(val_labels, 1)])
                        predictions = np.concatenate([predictions, infer_val])
                        losses.append(loss_val)

                    val_acc = np.mean(np.equal(predictions, ground_truth))
                    val_loss = np.mean(np.array(losses))

                    _log("-VALID- {} done: loss={:.4}, acc={:.4}"
                        .format(datetime.datetime.now(), val_loss, val_acc))

                    summ.log({
                        "validation_loss": val_loss,
                        "validation_acc": val_acc
                    }, step)

                step += 1

        # Save at the last step
        saver.save(sess, os.path.join(FLAGS.chkpnt_dir, str(step)), global_step=step)


def main():
    train_dat = Data(FLAGS.data_dir, "train", 1,
                     no_categories=100, suffix=".inceptionv3.pool.npy")
    valid_dat = Data(FLAGS.data_dir, "valid", 1,
                     no_categories=100, suffix=".inceptionv3.pool.npy")
    summ = ScalarSummarizer(FLAGS.summ_dir, {
        "training_loss": "float32",
        "validation_loss": "float32",
        "validation_acc": "float32"
    })
    _train(
        train_dat,
        valid_dat,
        summ)


if __name__ == "__main__":
    main()
