import tensorflow as tf


class MNISTNN(object):
  def classify(self, logits):
    return tf.argmax(logits, 1)

  def loss(self, logits, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(losses)

  def train(self, loss, step, dsize):
    return tf.train.GradientDescentOptimizer(self._learning_rate).minimize(loss, global_step=step)

class MNISTSimple(MNISTNN):
  def __init__(self, learning_rate):
    self._learning_rate = learning_rate

  def logits(self, inputs):
    hidden = tf.layers.dense(inputs=inputs, units=100, activation=tf.nn.relu, name='hidden')
    logits = tf.layers.dense(inputs=hidden, units=10, name='logit')
    return logits


class MNISTCNN(MNISTNN):
  def __init__(self, learning_rate):
    self._learning_rate = learning_rate

  def _weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def _bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def _conv2d(self, inputs, weight):
    return tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME')

  def _max_pool_2x2(self, inputs):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def logits(self, inputs):
    sqinputs = tf.reshape(inputs, [-1, 28, 28, 1])

    # First convolution layer
    W_conv1 = self._weight_variable([5, 5, 1, 32])
    b_conv1 = self._bias_variable([32])
    h_conv1 = tf.nn.relu(self._conv2d(sqinputs, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # Second convolution layer
    W_conv2 = self._weight_variable([5, 5, 32, 64])
    b_conv2 = self._bias_variable([64])
    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # Fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.layers.dense(h_pool2_flat, units=1024, activation=tf.nn.relu)

    # Dropout layer
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    logits = tf.layers.dense(h_fc1_drop, units=10)
    return logits