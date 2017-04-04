"""
An interface for NN models.
"""
import os

import numpy as np
import tensorflow as tf


class NN:
    """
    """

    def __init__(self, variables, name=""):
        """
        Initialize the model with variables.

        Params:
            variables (dict): A dictionary of variables.
        """
        self._variables = variables
        self._name = name

    def _get_file_name(self, directory):
        return os.path.join(directory,
                            "{}-{}.npz".format(self.__class__.__name__, self._name))

    def save(self, sess, directory):
        """
        Save the current model as an npz file.

        Params:
            sess (tf.Session()): The current session.
            directory: Directory in which this file should be saved.
        """
        def _eval_variables():
            return {key: self._variables[key].eval(sess)
                    for key in self._variables.keys()}

        # Save all variables as an npz file
        np.savez(self._get_file_name(directory), **_eval_variables())

    def restore(self, sess, directory):
        """
        Restore the current model from a saved file.

        Params:
            sess (tf.Session()): The current session.
            directory: Directory in which the weight file was saved.
        """
        def _assign_variables(values):
            for key in self._variables.keys():
                sess.run(self._variables[key].assign(values[key]))

        # Restore all variables from an npz file
        values = np.load(self._get_file_name(directory))
        _assign_variables(values)

    def outputs(self, inputs):
        pass
