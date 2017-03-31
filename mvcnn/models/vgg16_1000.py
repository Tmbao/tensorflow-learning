"""
A VGG16 implementation.
"""
import tensorflow as tf
import numpy as np

from nn import NN


class VGG16(NN):
    """
    A CNN model consisting of 16 layers.

    This model takes a 224x224 image of 3 channels [224x224x3] as an input and
    produces an output vector of 1000 dimensions.
    """
    @staticmethod
    def create_variables(from_file=None, trainable=True):
        """
        Create all variables for VGG16.

        Args:
            from_file: A file that stores weights.
        Returns:
            A dictionary of variables.
        """
        def _create_new_variables():
            def _weight_variable(shape):
                init = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(init)

            def _bias_variable(shape):
                init = tf.constant(0.1, shape=shape)
                return tf.Variable(init)

            return _weight_variable, _bias_variable

        def _load_variables():
            data = np.load(from_file)
            class Local:
                keys = sorted(data.keys())
                idx = 0

            def _weight_variable(shape):
                init = tf.constant(data[Local.keys[Local.idx]])
                Local.idx += 1
                return tf.Variable(init, trainable=trainable)

            def _bias_variable(shape):
                init = tf.constant(data[Local.keys[Local.idx]])
                Local.idx += 1
                return tf.Variable(init, trainable=trainable)

            return _weight_variable, _bias_variable


        variables = {}

        if from_file == None:
            _weight_variable, _bias_variable = _create_new_variables()
        else:
            _weight_variable, _bias_variable = _load_variables()

        # First group: conv + relu, conv + relu, pooling
        variables["11K"] = _weight_variable([3, 3, 3, 64])
        variables["11b"] = _bias_variable([64])
        variables["12K"] = _weight_variable([3, 3, 64, 64])
        variables["12b"] = _bias_variable([64])

        # Second group: conv + relu, conv + relu, pooling
        variables["21K"] = _weight_variable([3, 3, 64, 128])
        variables["21b"] = _bias_variable([128])
        variables["22K"] = _weight_variable([3, 3, 128, 128])
        variables["22b"] = _bias_variable([128])

        # Third group: conv + relu, conv + relu, conv + relu, pooling
        variables["31K"] = _weight_variable([3, 3, 128, 256])
        variables["31b"] = _bias_variable([256])
        variables["32K"] = _weight_variable([3, 3, 256, 256])
        variables["32b"] = _bias_variable([256])
        variables["33K"] = _weight_variable([3, 3, 256, 256])
        variables["33b"] = _bias_variable([256])

        # Forth group: conv + relu, conv + relu, conv + relu, pooling
        variables["41K"] = _weight_variable([3, 3, 256, 512])
        variables["41b"] = _bias_variable([512])
        variables["42K"] = _weight_variable([3, 3, 512, 512])
        variables["42b"] = _bias_variable([512])
        variables["43K"] = _weight_variable([3, 3, 512, 512])
        variables["43b"] = _bias_variable([512])

        # Fifth group : conv + relu, conv + relu, conv + relu, pooling
        variables["51K"] = _weight_variable([3, 3, 512, 512])
        variables["51b"] = _bias_variable([512])
        variables["52K"] = _weight_variable([3, 3, 512, 512])
        variables["52b"] = _bias_variable([512])
        variables["53K"] = _weight_variable([3, 3, 512, 512])
        variables["53b"] = _bias_variable([512])

        # 3 fully connected
        variables["6W"] = _weight_variable([7 * 7 * 512, 4096])
        variables["6b"] = _bias_variable([4096])
        variables["7W"] = _weight_variable([4096, 4096])
        variables["7b"] = _bias_variable([4096])
        variables["8W"] = _weight_variable([4096, 1000])
        variables["8b"] = _weight_variable([1000])

        return variables


    @staticmethod
    def create_model(variables=create_variables.__func__(), name=""):
        """
        Create an instance of VGG16.

        Params:
            variables (dict): A dictionary of variables.
        Returns:
            An instance of VGG16.
        """
        return VGG16(variables, name)


    def forward(self, inputs):
        """
        Get an output tensor corresponding to the input tensor.

        Params:
            inputs: An image of 224x224x3 or an equivalent tensor.
        Returns:
            A tensor of [1000].
        """
        def _apply_convolution(inputs, kernel, bias):
            return tf.nn.relu(
                tf.nn.conv2d(
                    inputs,
                    kernel,
                    strides=[1, 1, 1, 1],
                    padding="SAME") + bias)

        def _apply_pooling(inputs):
            return tf.nn.max_pool(
                inputs,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME")

        def _apply_fc_relu(inputs, weight, bias):
            return tf.nn.relu(tf.matmul(inputs, weight) + bias)

        def _get_variable(name):
            return self._variables[name]

        conv11 = _apply_convolution(inputs, 
                                    _get_variable("11K"),
                                    _get_variable("11b")) 
        conv12 = _apply_convolution(conv11,
                                    _get_variable("12K"),
                                    _get_variable("12b"))
        pool1 = _apply_pooling(conv12)

        conv21 = _apply_convolution(pool1,
                                    _get_variable("21K"),
                                    _get_variable("21b"))
        conv22 = _apply_convolution(conv21,
                                    _get_variable("22K"),
                                    _get_variable("22b"))
        pool2 = _apply_pooling(conv22)

        conv31 = _apply_convolution(pool2,
                                    _get_variable("31K"),
                                    _get_variable("31b"))
        conv32 = _apply_convolution(conv31,
                                    _get_variable("32K"),
                                    _get_variable("32b"))
        conv33 = _apply_convolution(conv32,
                                    _get_variable("33K"),
                                    _get_variable("33b"))
        pool3 = _apply_pooling(conv33)

        conv41 = _apply_convolution(pool3,
                                    _get_variable("41K"),
                                    _get_variable("41b"))
        conv42 = _apply_convolution(conv41,
                                    _get_variable("42K"),
                                    _get_variable("42b"))
        conv43 = _apply_convolution(conv42,
                                    _get_variable("43K"),
                                    _get_variable("43b"))
        pool4 = _apply_pooling(conv43)

        conv51 = _apply_convolution(pool4,
                                    _get_variable("51K"),
                                    _get_variable("51b"))
        conv52 = _apply_convolution(conv51,
                                    _get_variable("52K"),
                                    _get_variable("52b"))
        conv53 = _apply_convolution(conv52,
                                    _get_variable("53K"),
                                    _get_variable("53b"))
        pool5 = _apply_pooling(conv53)

        flat = tf.reshape(pool5, [-1, 7 * 7 * 512])

        fc1 = _apply_fc_relu(flat,
                             _get_variable("6W"),
                             _get_variable("6b"))
        fc2 = _apply_fc_relu(fc1,
                             _get_variable("7W"),
                             _get_variable("7b"))
        fc3 = _apply_fc_relu(fc2,
                             _get_variable("8W"),
                             _get_variable("8b"))

        return fc3
