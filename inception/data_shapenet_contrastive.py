"""
Data provider for trainer.

This data can be downloaded from https://sites.google.com/view/shrec17/dataset.
It contains views extracted from cad objects.
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

    def __init__(self, prefix, tag, no_categories=1000, suffix="", is_test=False, filter_fn=None):
        """
        Construct a data provider object.

        Params:
            prefix (str): Path to data directory.
            tag (str): Either train or valid.
        """
        def _initialize_labels():
            pcate = os.path.join(prefix, "all_cates.txt")
            with open(pcate, "r") as fcate:
                cates = [line.strip() for line in fcate.readlines()]
            cates = list(set(cates))
            cates = cates[1:]
            cate2id = {int(key): value for value, key in enumerate(cates)}

            plabel = os.path.join(prefix, tag + ".csv")
            label2cate = {}
            with open(plabel, "r") as flabel:
                reader = csv.reader(flabel)
                firstln = True
                for row in reader:
                    if firstln:
                         firstln = False
                         continue
                    label2cate[row[0]] = row[1]
            
            return {int(key): cate2id[int(value)] for key, value in label2cate.items()}

        self._objects = self._get_all_files(os.path.join(prefix, tag), suffix="")
        if filter_fn != None:
            self._objects = list(filter(filter_fn, self._objects))

        self._is_test = is_test
        if not is_test:
            self._label2id = _initialize_labels()

        self._no_categories = no_categories
        self._size = len(self._objects)
        
        paths = self._objects
        self._objects = []
        self._categories = []
        for path in paths:
            files = self._get_all_files(path, suffix=suffix)
            if not self._is_test:
                category = os.path.basename(path)
                logits[self._label2id[int(category)]] = 1.0
                self._categories += [category for _ in files]
            self._objects += files
        self._categories = np.array(self._categories)


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
        self._origin_objects = self._objects
        self._origin_catetories = self._categories
        self._objects = self._objects[idx]
        self._categories = self._categories[idx]

    def batches(self, batch_size, no_examples=-1):
        """
        Iterate over examples by batches
        """
        def _load_image(path):
            if self._suffix.lower() in [".jpg"]:
                return cv2.imread(path)
            else:
                return np.load(path)

        if no_examples == -1:
            no_examples = self._size
        else:
            no_examples = min(no_examples, self._size)

        for start in range(0, no_examples, batch_size):
            yield _load_image(self._origin_objects[start: min(no_examples, start +
                                                              batch_size)]),
                  self._origin_catetories[start: min(no_examples, start + batch_size)],
                  _load_image(self._objects[start: min(no_examples, start +
                                                       batch_size)]),
                  self.catetories[start: min(no_examples, start + batch_size)]

    def _get_all_files(self, path, suffix):
        return np.array(sorted([os.path.join(path, name)
                                for name in os.listdir(path)
                                if not name.startswith(".") and
                                not name.startswith("labels") and
                                name.lower().endswith(suffix)]))
