import os
import tensorflow as tf


class Saver:
  def __init__(self, chkpnt_dir, max_to_keep):
    self._chkpnt_path = os.path.join(chkpnt_dir, 'model.ckpt')
    self._max_to_keep = max_to_keep
    
  def init(self):
    self._saver = tf.train.Saver(tf.all_variables(), max_to_keep=self._max_to_keep)

  def save(self, sess, step):
    self._saver.save(sess, self._chkpnt_path, global_step=step)