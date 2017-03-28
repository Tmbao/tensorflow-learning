import tensorflow as tf
import numpy as np

from nn import NN


class Aggregator(NN):
    """
    A Aggregator consisting of n layers.
        - The first n - 1 layers are conv layers
        - The last layer is a fc layer.
    """
    @staticmethod
    def create_variables(dims=[7, 7, 512, 16], filter_threshold=32, from_file=None):
        """
        Create all variables for Aggregator.

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
            
        layer_id = 1

        # Create convolution layers to reduce the number of filters
        width, height, no_filters, no_classes = dims
        while (no_filters > filter_threshold):
            variables["{}K".format(layer_id)] = _weight_variable([3, 3, no_filters, no_filters / 2])
            variables["{}b".format(layer_id)] = _bias_variable([no_filters / 2])
            layer_id += 1
            no_filters /= 2

        # Create a fully connected layer for SVM purpose
        variables["{}W".format(layer_id)] = _weight_variable([width * height * no_filters, no_classes])
        variables["{}b".format(layer_id)] = _bias_variable([no_classes])

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
        return Aggregator(variables, name)


    def forward(self, inputs):
        """
        Get an output tensor corresponding to the input tensor.
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
                    strides=[1, 1, 1, 1],
                    padding="SAME")

        def _apply_fc_relu(inputs, weight, bias):
            return tf.nn.relu(tf.matmul(inputs, weight) + bias)

        def _get_variable(name):
            return self._variables[name]

        result = inputs
        no_layers = len(self._variables) / 2
        for i in range(1, no_layers):
            # Convolution
            result = _apply_convolution(result, 
                _get_variable("{}K".format(i)), 
                _get_variable("{}b".format(i)))
            # Pooling
            result = _apply_pooling(result)

        dims = result.get_shape().as_list()[1:]
        result = tf.reshape(result, [-1, reduce(lambda x, y: x * y, dims, 1)])

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        result = tf.nn.dropout(result, keep_prob)

        result = _apply_fc_relu(result,
            _get_variable("{}W".format(no_layers)),
            _get_variable("{}b".format(no_layers)))

        return result
