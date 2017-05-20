# Contrastive Loss
# by Che-Wei Lin
# Under the Simplified BSD License

import tensorflow as tf
from tensorflow.python.framework.function import Defun


def contrastive_loss(margin, threshold=1e-5):
    """Contrastive loss:
           E = sum(yd^2 + (1-y)max(margin-d, 0)^2) / 2 / N
           d = L2_dist(data1, data2)
       Usage:
           loss = contrastive_loss(1.0)(data1, data2, similarity)
       Note:
           This is a numeric stable version of contrastive loss
    """
    @Defun(tf.float32, tf.float32, tf.float32, tf.float32)
    def backward(data1, data2, similarity, diff):
        with tf.op_scope([data1, data2, similarity], "ContrastiveLoss_grad", "ContrastiveLoss_grad"):
            d_ = data1 - data2
            d_square = tf.reduce_sum(tf.square(d_), 1)
            d = tf.sqrt(d_square)

            minus = margin - d
            right_diff = minus / (d + threshold)
            right_diff = d_ * tf.reshape(right_diff * tf.to_float(tf.greater(minus, 0)), [-1, 1])

            batch_size = tf.to_float(tf.slice(tf.shape(data1), [0], [1]))
            data1_diff = diff * ((d_ + right_diff) * tf.reshape(similarity, [-1, 1]) - right_diff) / batch_size
            data2_diff = -data1_diff
            return data1_diff, data2_diff, tf.zeros_like(similarity)

    @Defun(tf.float32, tf.float32, tf.float32, grad_func=backward)
    def forward(data1, data2, similarity):       # assume similarity shape = (N,)
        with tf.op_scope([data1, data2, similarity], "ContrastiveLoss", "ContrastiveLoss"):
            d_ = data1 - data2
            d_square = tf.reduce_sum(tf.square(d_), 1)
            d = tf.sqrt(d_square)

            minus = margin - d
            sim = similarity * d_square
            nao = (1.0 - similarity) * tf.square(tf.maximum(minus, 0))
            return tf.reduce_mean(sim + nao) / 2

    return forward