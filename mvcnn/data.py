import os
import re
import cv2
import numpy as np


class Data:
    """
    Data provider for tensorflow.
    """
    def __init__(self, prefix, tag, no_views=26, no_categories=1000):
        """
        Construct a data provider object.

        Params:
            prefix (str): Path to data directory.
            tag (str): Either train or valid.
        """
        def _initialize_labels(path):
            path = os.path.join(path, "labels.txt")
            with open(path, "r") as f:
                labels = [line.strip() for line in f.readlines()]
            labels = list(set(labels))
            label2id = {key: value for value, key in enumerate(labels)}
            return label2id

        self._objects = self._get_all_files(os.path.join(prefix, tag))
        self._label2id = _initialize_labels(os.path.join(prefix, tag))
        self._size = len(self._objects)
        self._no_views = no_views
        self._no_categories = no_categories

       
    def size(self):
        """
        """
        return self._size


    def shuffle(self):
        """
        """
        idx = range(self._size)
        np.random.shuffle(idx)
        self._objects = self._objects[idx]


    def batches(self, batch_size, no_examples=-1):
        """
        """
        if no_examples == -1:
            no_examples = self._size
        else:
            no_examples = min(no_examples, self._size)

        for start in xrange(0, no_examples, batch_size):
            yield self._load_objects(self._objects[start : min(no_examples, start +
                batch_size)])


    def _get_all_files(self, path):
        return np.array(sorted([os.path.join(path, name) 
            for name in os.listdir(path) 
            if not name.startswith(".") and not name.startswith("labels")]))

     
    def _load_objects(self, objects):
        def _load_image(path):
            return cv2.imread(path)

        inputs = [[] for _ in xrange(self._no_views)]
        labels = []
        for obj in objects:
            # Get labels
            category = re.search("\/([a-z]+)\_", obj).group(1)
            logits = [0.0] * self._no_categories
            logits[self._label2id[category]] = 1.0
            labels.append(logits)
        
            views = self._get_all_files(obj)
            for i in xrange(self._no_views):
                inputs[i].append(_load_image(views[i]))

        return np.array(inputs), np.array(labels)


