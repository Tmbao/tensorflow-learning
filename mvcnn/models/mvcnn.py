import tensorflow as tf
import numpy as np


class MVCNN:
    """
    A multi-view CNN for recognizing 3d objects.
    """

    def __init__(self, 
            view_create,
            view_vars,
            aggr_create,
            aggr_vars,
            no_views=26):
        """
        Construct a MVCNN.

        Params:
            cnn1_creator: A function that creates an instance of cnn1.
            cnn1_vars (dict): Variables for cnn1.
            cnn2_creator: A function that creates an instnace of cnn2.
            cnn2_vars (dict): Variables for cnn2.
            no_views: Number of views.
        """
        self._no_views = no_views
        self._views = [view_create(view_vars, name="view") for _ in xrange(no_views)]
        self._aggr = aggr_create(aggr_vars, name="aggr")


    def save(self, sess, directory):
        """
        Save the current model as an npz file.

        Params:
            sess (tf.Session()): The current session.
            directory: Directory in which this file should be saved.
        """
        self._views[0].save(sess, directory)
        self._aggr.save(sess, directory)


    def restore(self, sess, directory):
        """
        Restore the current model from a saved file.

        Params:
            sess (tf.Session()): The current session.
            directory: Directory in which the weight file was saved.
        """
        self._view[0].restore(sess, directory)
        self._aggr.save(sess, directory)


    def outputs(self, inputs):
        """
        Get an output tensor for inputs.

        Params:
            inputs: An array of no_views elements where each element is an image
            of 224x224x3 or an equivalent tensor.
        Returns:
            An output tensor.
        """
        view_outputs = [self._views[i].outputs(inputs[i]) 
                for i in xrange(self._no_views)]
        concatenated_outputs = tf.concat([tf.expand_dims(view_output, 0) 
            for view_output in view_outputs], 0)
        view_pooling = tf.reduce_max(concatenated_outputs, [0])
        return self._aggr.outputs(view_pooling)
