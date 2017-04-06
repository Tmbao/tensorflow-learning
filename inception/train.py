#!/usr/local/bin/python3
"""
Training program.
"""
import datetime
import os

import numpy as np
import tensorflow as tf

from data import Data
from nets.fc_aggregator import FCAggregator
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
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
tf.app.flags.DEFINE_integer("log_period", 5, "Log period")
tf.app.flags.DEFINE_integer("val_period", 25, "Validation period")
tf.app.flags.DEFINE_integer("save_period", 200, "Saving period")
tf.app.flags.DEFINE_integer("no_epoch", 1000, "Number of epoches")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")


def _log(message):
    if FLAGS.verbose:
        print(message)


def _get_loss_op(logits, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
    return tf.reduce_mean(losses)


def _get_train_op(loss_op, step, learning_rate):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss_op, global_step=step)


def _get_infer_op(logits):
    return tf.argmax(logits, 1)


def _save_model(nn, sess, saving_dir):
    _log("-SAVE- start")
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    nn.save(sess, saving_dir)
    _log("-SAVE- done")


def _train(
        train_dat,
        valid_dat,
        summ,
        chkpnt_dir,
        from_step,
        batch_size,
        learning_rate,
        log_period,
        val_period,
        save_period,
        no_epoch):

    train_size = train_dat.size()
    _log("training size = {}".format(train_size))
    _log("validation size = {}".format(valid_dat.size()))

    # Create a new graph
    graph = tf.Graph()
    with graph.as_default():
        # Update step
        step = from_step + 1

        # Start a session
        sess = tf.Session(graph=graph)

        _log("initializing the model")

        # Place holders
        inputs = tf.placeholder("float32", shape=(None, 1008))
        labels = tf.placeholder("float32", shape=(None, 16))

        # Tensors
        aggr = FCAggregator.create_model(variables=FCAggregator.create_variables(
            graph=graph, dims=[1008, 512, 16]), name="aggr")
        forward_op = aggr.forward(inputs)
        loss_op = _get_loss_op(forward_op, labels)
        train_op = _get_train_op(loss_op, tf.Variable(step, trainable=False),
                                 learning_rate)
        infer_op = _get_infer_op(forward_op)

        sess.run(tf.global_variables_initializer())
        if from_step >= 0:
            _log("restoring the model")
            aggr.restore(sess, os.path.join(chkpnt_dir, str(from_step)))

        for epoch in range(no_epoch):
            _log("{} epoch = {}".format(datetime.datetime.now(), epoch))
            train_dat.shuffle()

            for trn_inputs, trn_labels, _ in train_dat.batches(batch_size):
                trn_inputs = np.squeeze(trn_inputs)

                # Create food
                food = {labels: trn_labels, inputs: trn_inputs}

                # Feed the model
                _, _, loss_val = sess.run([train_op, infer_op, loss_op],
                                          feed_dict=food)

                summ.log({"training_loss": loss_val}, step)

                # Log info
                if step % log_period == 0:
                    _log("-TRAIN- {} step={}, loss={:.4}"
                         .format(datetime.datetime.now(), step, loss_val))

                # Save the current model
                if step % save_period == 0:
                    _save_model(aggr, sess, os.path.join(
                        chkpnt_dir, str(step)))

                # Perform validation
                if step > 0 and step % val_period == 0:
                    _log("-VALID- {} start".format(datetime.datetime.now()))
                    valid_dat.shuffle()
                    ground_truth = []
                    predictions = []
                    losses = []
                    for val_inputs, val_labels, _ in valid_dat.batches(batch_size):
                        val_inputs = np.squeeze(val_inputs)

                        # Create food
                        food = {labels: val_labels, inputs: val_inputs}

                        infer_val, loss_val, = sess.run(
                            [infer_op, loss_op], feed_dict=food)

                        ground_truth = np.concatenate(
                            [ground_truth, np.argmax(val_labels, 1)])
                        predictions = np.concatenate([predictions, infer_val])
                        losses.append(loss_val)

                    val_acc = np.mean(
                        np.equal(predictions[:len(ground_truth)], ground_truth))
                    val_loss = np.mean(np.array(losses))

                    _log("-VALID- {} done: loss={:.4}, acc={:.4}"
                        .format(datetime.datetime.now(), val_loss, val_acc))

                    summ.log({
                        "validation_loss": val_loss,
                        "validation_acc": val_acc
                    }, step)

                step += 1

        # Save at the last step
        _save_model(aggr, sess, os.path.join(chkpnt_dir, str(step)))


def main():
    train_dat = Data(FLAGS.data_dir, "train", 1,
                     no_categories=16, suffix=".npy")
    valid_dat = Data(FLAGS.data_dir, "valid", 1,
                     no_categories=16, suffix=".npy")
    summ = ScalarSummarizer(FLAGS.summ_dir, {
        "training_loss": "float32",
        "validation_loss": "float32",
        "validation_acc": "float32"
    })
    _train(
        train_dat,
        valid_dat,
        summ,
        FLAGS.chkpnt_dir,
        FLAGS.from_step,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.log_period,
        FLAGS.val_period,
        FLAGS.save_period,
        FLAGS.no_epoch)


if __name__ == "__main__":
    main()
