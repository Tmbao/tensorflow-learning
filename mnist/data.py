import os
import numpy as np



class Data:
    def __init__(self, prefix, tag):
        self._images = self._load_images(os.path.join(prefix, tag, 'images'))
        self._labels = self._load_labels(os.path.join(prefix, tag, 'labels'))
        self._size = self._images.shape[0]

    def _to_int(self, bytes):
        return reduce(lambda acc, x: (acc << 8) + x, bytes)

    def _load_images(self, path):
        data = np.fromfile(path, dtype='u1')
        n_images = self._to_int(data[4:8].tolist())
        n_rows = self._to_int(data[8:12].tolist())
        n_cols = self._to_int(data[12:16].tolist())
        return data[16:].reshape(n_images, n_rows * n_cols).astype(float)

    def _load_labels(self, path):
        data = np.fromfile(path, dtype='u1')
        n_labels = self._to_int(data[4:8].tolist())
        labels = np.zeros([n_labels, 10])
        labels[range(n_labels), data[8:].tolist()] = 1
        return labels

    def size(self):
        return self._size

    def shuffle(self):
        idx = range(self._size)
        np.random.shuffle(idx)

        self._images = self._images[idx]
        self._labels = self._labels[idx]

    def batches(self, batch_size):
        for start in xrange(0, self._size, batch_size):
            yield self._images[start : start + batch_size], self._labels[start : start + batch_size]

    def images(self):
        return self._images

    def labels(self):
        return self._labels

