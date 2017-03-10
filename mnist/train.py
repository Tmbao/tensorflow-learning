import tensorflow as tf
import numpy as np
from datetime import datetime

from model import MNISTSimple, MNISTCNN
from data import Data
from saver import Saver
from summarizer import ScalarSummarizer


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '', 'Data directory')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_string('chkpnt_dir', '', 'Check-point directory')
tf.app.flags.DEFINE_string('from_chkpnt', None, 'Start training from a \
        checkpoint')
tf.app.flags.DEFINE_string('summary_dir', '', 'Summary directory')
tf.app.flags.DEFINE_integer('max_to_keep', 100, 'Maximum number of checkpoints')
tf.app.flags.DEFINE_boolean('verbose', False, 'Verbose')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.app.flags.DEFINE_integer('log_period', 10, 'Log period')
tf.app.flags.DEFINE_integer('val_period', 200, 'Validation period')
tf.app.flags.DEFINE_integer('save_period', 500, 'Validation period')


def log(message):
    if FLAGS.verbose:
        print message


def train(train_data,
          valid_data,
          saver,
          summ,
          from_chkpnt,
          batch_size,
          learning_rate,
          log_period,
          val_period,
          save_period):

    train_data.shuffle()
    valid_data.shuffle()

    train_size = train_data.size()
    log('training size = %d' % train_size)

    with tf.Graph().as_default():
        start_step = 0

        images = tf.placeholder('float32', shape=(None, 784), name='image')
        labels = tf.placeholder('float32', shape=(None, 10), name='label')

        # Declare some tensors
        mnist = MNISTCNN(learning_rate)
        logits = mnist.logits(images)
        loss = mnist.loss(logits, labels)
        train_op = mnist.train(loss, tf.Variable(start_step, trainable=False), train_size)
        classify = mnist.classify(logits)

        # Create a new session
        sess = tf.Session()

        if from_chkpnt == None:
            log('initializing the model')
            sess.run(tf.global_variables_initializer())
            saver.init()
        else:
            log('loading the latest checkpoint')
            saver.restore(sess, from_chkpnt)


        test_val = tf.Variable(tf.constant(0, shape=[2]))
        sess.run(test_val.assign(np.array([1, 2])))
        with sess.as_default():
            print test_val.eval().type

        # Start training
        step = start_step
        for epoch in xrange(20):
            log('epoch = %d' % epoch)

            for b_images, b_labels in train_data.batches(batch_size):

                food = {images: b_images, labels: b_labels, 'keep_prob:0': 0.5}
                _, pred_vals, loss_val = sess.run([train_op, classify, loss], feed_dict=food)

                # Print out information of the current step
                if step % log_period == 0:
                    log('TRAIN: step=%d, loss=%.2f' % (step, loss_val))

                # Validate the model
                if step % val_period == 0:
                    val_expects = []
                    val_predicts = []
                    val_losses = []
                    for val_images, val_labels in valid_data.batches(batch_size):
                        v_expects = np.argmax(val_labels, 1)
                        v_predicts, v_loss, = sess.run([classify, loss],
                                feed_dict={
                                    images: val_images, 
                                    labels: val_labels,
                                    'keep_prob:0': 1})
                        
                        val_expects = np.concatenate([val_expects, v_expects])
                        val_predicts = np.concatenate([val_predicts, v_predicts])
                        val_losses.append(v_loss)

                    val_corrects = np.equal(val_predicts[:len(val_expects)], val_expects)
                    val_acc = np.mean(val_corrects)
                    val_loss = np.mean(np.array(val_losses))

                    log('VALIDATION: loss=%.2f, acc=%.2f' % (val_loss, val_acc))

                    summ.log({
                        'validation_loss': val_loss,
                        'validation_accuracy': val_acc
                        }, step)

                # Save the model
                if step % save_period == 0:
                    saver.save(sess, step=step)

                step += 1


def main():
    train_data = Data(FLAGS.data_dir, tag='train')
    valid_data = Data(FLAGS.data_dir, tag='valid')
    train(train_data, valid_data, 
        Saver(FLAGS.chkpnt_dir, max_to_keep=FLAGS.max_to_keep),
        ScalarSummarizer(FLAGS.summary_dir, {
            'validation_loss': 'float32',
            'validation_accuracy': 'float32'}),
        FLAGS.from_chkpnt,
        FLAGS.batch_size, 
        FLAGS.learning_rate,
        FLAGS.log_period,
        FLAGS.val_period,
        FLAGS.save_period)


if __name__ == '__main__':
    main()
