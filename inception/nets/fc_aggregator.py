"""
An SVM class using fully-connected layers.
"""
import numpy as np
import tensorflow as tf

from nets.nn import NN


class FCAggregator(NN):
    """
    A FCAggregator consisting of n relu-ed fully connected layers.
    """
    @staticmethod
    def create_variables(dims=[1008, 512, 16], from_file=None):
        """
        Create all variables for FCAggregator.

        Args:
            dims: An array of the number of dimensions each layer, dims[0]
            corresponds to input layer and dims[-1] corresponds to output layer.
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
            keys = sorted(data.keys())
            idx = 0

            def _weight_variable(shape):
                init = tf.constant(data[keys[idx]])
                idx += 1
                return tf.Variable(init)

            def _bias_variable(shape):
                init = tf.constant(data[keys[idx]])
                idx += 1
                return tf.Variable(init)

            return _weight_variable, _bias_variable

        variables = {}

        if from_file == None:
            _weight_variable, _bias_variable = _create_new_variables()
        else:
            _weight_variable, _bias_variable = _load_variables()

        no_layers = len(dims)
        for i in range(1, no_layers):
            variables["{}W".format(i)] = _weight_variable([dims[i - 1],
                                                           dims[i]])
            variables["{}b".format(i)] = _bias_variable([dims[i]])

        return variables

    @staticmethod
    def create_model(variables=create_variables.__func__(), name=""):
        """
        Create an instance of AggregatorNN.

        Params:
            variables (dict): A dictionary of variables.
        Returns:
            An instance of AggregatorNN.
        """
        return FCAggregator(variables, name)

    def forward(self, inputs):
        result = inputs
        no_layers = len(self._variables) // 2
        for i in range(1, no_layers + 1):
            W = self._variables["{}W".format(i)]
            b = self._variables["{}b".format(i)]
            result = tf.nn.relu(tf.matmul(result, W) + b)

        return result
