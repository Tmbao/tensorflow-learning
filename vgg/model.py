import tensorflow as tf


class VGG16(object):

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    def logits(self, inputs):
        """
        Construct a model on top of an input tensor. 

        Args:
            inputs (tensor): An input tensor/placeholder of 224x224x3.
        Returns:
            tensor: The last layer of the constructed model.
        """
    
        def _weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def _bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def _conv2d(inputs, weight):
            return tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME')

        def _apply_convolution(inputs, kernel_size):
            k_conv = _weight_variable(kernel_size)
            b_conv = _bias_variable([kernel_size[3]])
            conv = tf.nn.relu(_conv2d(inputs, k_conv) + b_conv)
            return conv

        def _apply_pooling(inputs):
            return tf.nn.max_pool(inputs, 
                    ksize=[1, 2, 2, 1], 
                    strides=[1, 2, 2, 1],
                    padding='SAME')

        def _apply_fc_relu(inputs, units):
            return tf.layers.dense(inputs=inputs, units=units,
                    activation=tf.nn.relu)

        # 1st group
        conv1_1 = _apply_convolution(inputs, [3, 3, 3, 64])
        conv1_2 = _apply_convolution(conv1_1, [3, 3, 64, 64])
        pool1 = _apply_pooling(conv1_2)

        # 2nd group
        conv2_1 = _apply_convolution(pool1, [3, 3, 64, 128])
        conv2_2 = _apply_convolution(conv2_1, [3, 3, 128, 128])
        pool2 = _apply_pooling(conv2_2)

        # 3rd group
        conv3_1 = _apply_convolution(pool2, [3, 3, 128, 256])
        conv3_2 = _apply_convolution(conv3_1, [3, 3, 256, 256])
        conv3_3 = _apply_convolution(conv3_2, [3, 3, 256, 256])
        pool3 = _apply_pooling(conv3_3)

        # 4th group
        conv4_1 = _apply_convolution(pool3, [3, 3, 256, 512])
        conv4_2 = _apply_convolution(conv4_1, [3, 3, 512, 512])
        conv4_3 = _apply_convolution(conv4_2, [3, 3, 512, 512])
        pool4 = _apply_pooling(conv4_3)

        # 5th group
        conv5_1 = _apply_convolution(pool4, [3, 3, 512, 512])
        conv5_2 = _apply_convolution(conv5_1, [3, 3, 512, 512])
        conv5_3 = _apply_convolution(conv5_2, [3, 3, 512, 512])
        pool5 = _apply_pooling(conv5_3)

        # Flatten 
        flattened_pool5 = tf.reshape(pool5, [-1, 7 * 7 * 512])

        # 1st fully-connected
        fc1 = _apply_fc_relu(flattened_pool5, 4096)
        fc2 = _apply_fc_relu(fc1, 4096)
        fc3 = _apply_fc_relu(fc2, 1000)
        
        return fc3 

    def infer(self, logits):
        return tf.argmax(logits, 1)

    def loss(self, logits, labels):
        losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits)
        return tf.reduce_mean(losses)

    def train(self, loss, step):
        return tf.train.GradientDescentOptimizer(
                self._learning_rate).minimize(loss, global_step=step)
