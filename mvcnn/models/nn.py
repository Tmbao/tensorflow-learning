import tensorflow as tf
import numpy as np
import os


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
            with sess.as_default():
                result = {key: self._variables[key].eval()
                        for key in self._variables.keys()}
            return result

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
        values = np.load(self._get_file_name())
        _assign_variables(values) 


    def outputs(self, inputs):
        pass
