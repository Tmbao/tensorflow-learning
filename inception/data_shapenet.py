"""
Data provider for trainer.

This data can be downloaded from https://sites.google.com/view/shrec17/dataset.
It contains views extracted from 3d objects of irregular holes.
"""
import csv
import os
import re

import numpy as np

import cv2


class Data:
    """
    Data provider for tensorflow.
    """

    def __init__(self, prefix, tag, no_views=26, no_categories=1000, suffix=""):
        """
        Construct a data provider object.

        Params:
            prefix (str): Path to data directory.
            tag (str): Either train or valid.
        """
        def _initialize_labels():
            pcate = os.path.join(prefix, "all_cates.txt")
            with open(pcate, "r") as fcate:
                cates = [line.strip() for fcate in flabel.readlines()]
            cates = list(set(cates))
            cate2id = {key: value for value, key in enumerate(cates)}

            plabel = os.path.join(prefix, tag + ".csv")
            label2cate = {}
            with open(plabel, "r") as flabel:
                reader = csv.reader(flabel)
                for row in reader:
                    label2cate[row[0]] = row[1]
            
            return {key: cate2id[value] for key, value in label2cate.items}

        self._objects = self._get_all_files(
            os.path.join(prefix, tag), suffix="")
        self._label2id = _initialize_labels()
        self._size = len(self._objects)
        self._no_views = no_views
        self._no_categories = no_categories
        self._suffix = suffix

    def size(self):
        """
        Get the number of examples.
        """
        return self._size

    def shuffle(self):
        """
        Shuffle all examples.
        """
        idx = list(range(self._size))
        np.random.shuffle(idx)
        self._objects = self._objects[idx]

    def batches(self, batch_size, no_examples=-1):
        """
        Iterate over examples by batches
        """
        if no_examples == -1:
            no_examples = self._size
        else:
            no_examples = min(no_examples, self._size)

        for start in range(0, no_examples, batch_size):
            yield self._load_objects(self._objects[start: min(no_examples, start +
                                                              batch_size)])

    def _get_all_files(self, path, suffix):
        return np.array(sorted([os.path.join(path, name)
                                for name in os.listdir(path)
                                if not name.startswith(".") and
                                not name.startswith("labels") and
                                name.lower().endswith(suffix)]))

    def _load_objects(self, objects):
        def _load_image(path):
            if self._suffix.lower() in [".jpg"]:
                return cv2.imread(path)
            else:
                return np.load(path)

        inputs = [[] for _ in range(self._no_views)]
        labels = []
        for obj in objects:
            # Get labels
            category = os.path.basename(obj)
            logits = [0.0] * self._no_categories
            logits[self._label2id[category]] = 1.0
            labels.append(logits)
            views = self._get_all_files(obj, suffix=self._suffix)
            for i in range(self._no_views):
                inputs[i].append(_load_image(views[i]))

        return np.array(inputs), np.array(labels), objects
