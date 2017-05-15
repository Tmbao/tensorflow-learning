"""
Summarizer for trainer.
"""
import tensorflow as tf


class ScalarSummarizer:
    def __init__(self, summary_dir, sess, tensors):
        self._sess = sess
        self._writer = tf.summary.FileWriter(summary_dir)
        self._summary_tensors = {}
        for tensor_name, tensor_type in tensors.items():
            self._summary_tensors[tensor_name] = tf.summary.scalar(
                tensor_name,
                tf.placeholder(tensor_type, name=tensor_name))

    def log(self, tensors, step):
        for tensor_name, tensor_val in tensors.items():
            self._writer.add_summary(self._sess.run(
                self._summary_tensors[tensor_name],
                feed_dict={
                    tensor_name + ':0': tensor_val
                }), step)
