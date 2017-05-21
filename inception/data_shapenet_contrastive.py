"""
Data provider for trainer.

This data can be downloaded from https://sites.google.com/view/shrec17/dataset.
It contains views extracted from cad objects.
"""
import csv
import os
import random
import re

import numpy as np

import cv2


class Data:
    """
    Data provider for tensorflow.
    """

    def __init__(
            self,
            prefix,
            tag,
            no_categories=1000,
            suffix="",
            is_test=False,
            filter_fn=None):
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

            return {int(key): cate2id[int(value)]
                    for key, value in label2cate.items()}

        self._objects = self._get_all_files(
            os.path.join(prefix, tag), suffix="")
        if filter_fn is not None:
            self._objects = list(filter(filter_fn, self._objects))

        self._is_test = is_test
        if not is_test:
            self._label2id = _initialize_labels()

        self._no_categories = no_categories

        paths = self._objects
        self._objects = []
        self._categories = []
        for path in paths:
            files = self._get_all_files(path, suffix=suffix)
            if not self._is_test:
                category = os.path.basename(path)
                self._categories += [category for _ in files]
            self._objects.append(files)
        self._origin_objects = np.concatenate(self._objects)
        self._origin_categories = np.array(self._categories)
        self._suffix = suffix

        self._size = len(self._origin_objects)
        self._groups = [[] for _ in range(self._no_categories)]
        for idx in range(self._size):
            self._groups[self._categories[idx]].append(self._origin_objects[idx])

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
        self._objects = self._origin_objects[idx]
        self._categories = self._origin_categories[idx]

    def batches(self, batch_size, no_examples=-1, impostor=True):
        """
        Iterate over examples by batches
        """
        def _load_image(path):
            if self._suffix.lower() in [".jpg"]:
                return cv2.resize(cv2.imread(path), (299, 299))
            else:
                return np.load(path)

        def _load_images(paths):
            images = []
            for path in paths:
                images.append(_load_image(path))
            return np.array(images)

        def _load_similar_images(categories):
            similar_images = []
            for category in categories:
                group_size = len(self._groups[category])
                similar_images.append(self._groups[category][random.randint(0, group_size - 1)])
            return np.array(similar_images)

        def _load_logits(categories):
            logits = [0.0] * self._no_categories
            logits[self._label2id[int(category)]] = 1.0

        if no_examples == -1:
            no_examples = self._size
        else:
            no_examples = min(no_examples, self._size)

        for start in range(0, no_examples, batch_size):
            if impostor:
                yield (_load_images(self._origin_objects[start: min(no_examples, start + batch_size)]),
                       _load_logits(self._origin_categories[start: min(no_examples, start + batch_size)]),
                       _load_similar_images(self._origin_categories[start: min(no_examples, start + batch_size)]),
                       _load_logits(self._origin_categories[start: min(no_examples, start + batch_size)]))
            else:
                yield (_load_images(self._origin_objects[start: min(no_examples, start + batch_size)]),
                       _load_logits(self._origin_categories[start: min(no_examples, start + batch_size)]),
                       _load_images(self._objects[start: min(no_examples, start + batch_size)]),
                       _load_logits(self._categories[start: min(no_examples, start + batch_size)]))

    def _get_all_files(self, path, suffix):
        return np.array(sorted([os.path.join(path, name)
                                for name in os.listdir(path)
                                if not name.startswith(".") and
                                not name.startswith("labels") and
                                name.lower().endswith(suffix)]))
