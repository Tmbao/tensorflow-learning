import tensorflow as tf


class ScalarSummarizer:
    def __init__(self, summary_dir, tensors, graph=tf.Graph()):
        self._graph = graph
        with self._graph.as_default():
            self._sess = tf.Session()
            self._writer = tf.summary.FileWriter(summary_dir, graph=self._graph)
            self._summary_tensors = {}
            for tensor_name, tensor_type in tensors.iteritems():
                self._summary_tensors[tensor_name] = tf.summary.scalar(
                    tensor_name,
                    tf.placeholder(tensor_type, name=tensor_name))

    def log(self, tensors, step):
        with self._graph.as_default():
            for tensor_name, tensor_val in tensors.iteritems():
                self._writer.add_summary(self._sess.run(
                    self._summary_tensors[tensor_name],
                    feed_dict={
                        tensor_name + ':0': tensor_val
                    }), step)
