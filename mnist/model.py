import tensorflow as tf


class MNISTSimple:
  def __init__(self, learning_rate):
    self._learning_rate = learning_rate

  def logits(self, inputs):
    hidden = tf.layers.dense(inputs=inputs, units=100, activation=tf.nn.relu, name='hidden')
    logits = tf.layers.dense(inputs=hidden, units=10, name='logit')
    return logits


  def classify(self, logits):
    return tf.argmax(logits, 1)

  def loss(self, logits, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(losses)

  def train(self, loss, step, dsize):
    return tf.train.GradientDescentOptimizer(self._learning_rate).minimize(loss, global_step=step)
