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
tf.app.flags.DEFINE_string("chkpnt_dir", "{}/chkpnts".format(DEFAULT_DIR),
                           "Checkpoint directory")
tf.app.flags.DEFINE_string("summ_dir", "{}/summary".format(DEFAULT_DIR),
                           "Summary directory")
tf.app.flags.DEFINE_integer("no_views", 26, "Number of views")
tf.app.flags.DEFINE_integer("from_step", -1,
                            "Continue training from a checkpoint")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.000001, "Learning rate")
tf.app.flags.DEFINE_integer("log_period", 5, "Log period")
tf.app.flags.DEFINE_integer("val_period", 25, "Validation period")
tf.app.flags.DEFINE_integer("save_period", 50, "Saving period")
tf.app.flags.DEFINE_integer("no_epoch", 100, "Number of epoches")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")


def _log(message):
    if FLAGS.verbose:
        print(message)


def _loss(logits, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
    return tf.reduce_mean(losses)


def _trainer(loss, step, learning_rate):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=step)


def _infer(logits):
    return tf.argmax(logits, 1)


def _train(
        train_dat,
        valid_dat,
        summ,
        no_views,
        chkpnt_dir,
        from_step,
        batch_size,
        learning_rate,
        log_period,
        val_period,
        save_period,
        no_epoch):

    train_dat.shuffle()
    valid_dat.shuffle()

    train_size = train_dat.size()
    _log("training size = {}".format(train_size))

    step = from_step + 1

    # Start a session
    sess = tf.Session()

    # Place holders
    inputs = tf.placeholder("float32", shape=(None, 1008))
    labels = tf.placeholder("float32", shape=(None, 16))

    # Tensors
    cnn = InceptionV3()
    aggr = FCAggregator.create_model()
    outputs = aggr.forward(inputs)
    loss = _loss(outputs, labels)
    trainer = _trainer(loss, tf.Variable(step, trainable=False),
                       learning_rate)
    infer = _infer(outputs)

    _log("initializing the model")
    sess.run(tf.global_variables_initializer())
    if from_step > 0:
        _log("restoring the model")
        aggr.restore(sess, os.path.join(chkpnt_dir, str(from_step)))

    for epoch in range(no_epoch):
        _log("epoch = {}".format(epoch))

        for b_inputs, b_labels in train_dat.batches(batch_size):
            # Merge outputs from CNN
            concated_outputs = np.concatenate([np.expand_dims(cnn.classify(b_input), 0)
                                               for b_input in b_inputs], 0)
            view_pooling = np.amax(concated_outputs, 0)

            # Create food
            food = {labels: b_labels, inputs: view_pooling}

            # Feed the model
            _, _, loss_val = sess.run([trainer, infer, loss],
                                      feed_dict=food)

            # Log info
            if step % log_period == 0:
                _log("-TRAIN- step={}, loss={}".format(step, loss_val))

            # Save the current model
            if step % save_period == 0:
                _log("-SAVE- start")
                saving_dir = os.path.join(chkpnt_dir, str(step))
                if not os.path.exists(saving_dir):
                    os.makedirs(saving_dir)

                aggr.save(sess, saving_dir)
                _log("-SAVE- done")

            # Perform validation
            if step > 0 and step % val_period == 0:
                _log("-VALID- start")
                val_expects = []
                val_predicts = []
                val_losses = []
                for val_inputs, val_labels in valid_dat.batches(batch_size):
                    # Merge outputs from CNN
                    concated_outputs = np.concatenate([np.expand_dims(cnn.classify(v_input), 0)
                                                       for v_input in val_inputs], 0)
                    view_pooling = np.amax(concated_outputs, 0)

                    # Create food
                    food = {labels: b_labels, inputs: view_pooling}

                    v_expects = np.argmax(val_labels, 1)
                    v_predicts, v_loss, = sess.run([infer, loss],
                                                   feed_dict=food)

                    val_expects = np.concatenate([val_expects, v_expects])
                    val_predicts = np.concatenate(
                        [val_predicts, v_predicts])
                    val_losses.append(v_loss)

                val_corrects = np.equal(
                    val_predicts[:len(val_expects)], val_expects)
                val_acc = np.mean(val_corrects)
                val_loss = np.mean(np.array(val_losses))

                _log("-VALID- done: loss=%.2f, acc=%.2f" %
                     (val_loss, val_acc))

                summ.log({
                    "validation_loss": val_loss,
                    "validation_acc": val_acc
                }, step)

            step += 1


def main():
    train_dat = Data(FLAGS.data_dir, "train", FLAGS.no_views, no_categories=16)
    valid_dat = Data(FLAGS.data_dir, "valid", FLAGS.no_views, no_categories=16)
    summ = ScalarSummarizer(FLAGS.summ_dir, {
        "validation_loss": "float32",
        "validation_acc": "float32"
    })
    _train(
        train_dat,
        valid_dat,
        summ,
        FLAGS.no_views,
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
