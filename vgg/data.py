import os
import numpy as np
import cv2

try:
    import cPickle as pickle
except:
    import pickle


class Data:
    def __init__(self, prefix, tag, no_labels=1000, image_size=(3, 32, 32)):
        self._image_size = image_size
        self._no_labels = no_labels
        self._images = np.zeros((0, 3072))
        self._labels = []
        for f in self._get_all_files(prefix, tag):
            new_images, new_labels = self._load_batch(f)
            new_images = new_images
            new_labels = np.array(new_labels)

            self._images = np.concatenate([
                new_images,    
                self._images], axis=0)
            self._labels = np.concatenate([
                new_labels,
                self._labels], axis=0)
        self._size = len(self._images)
        
    def _get_all_files(self, prefix, tag):
        path = os.path.join(prefix, tag)
        files = [os.path.join(prefix, tag, f) for f in os.listdir(path) if not f.startswith('.')]
        return files

    def _load_batch(self, file_name):
        with open(file_name, 'rb') as fb:
            data = pickle.load(fb)
        return data['data'], data['labels']

    def _refine_images(self, images):
        sample_size = len(images)
        images = np.resize(images, [sample_size, 
            self._image_size[0], self._image_size[1], self._image_size[2]])
        refined_images = []
        for image in images:
            r, g, b = image
            r = cv2.resize(r, (224, 224))
            g = cv2.resize(g, (224, 224))
            b = cv2.resize(b, (224, 224))

            refined_images.append([r, g, b])
        refined_images = np.transpose(np.array(refined_images), [0, 2, 3, 1])
        return refined_images.astype(float)

    def _refine_labels(self, labels):
        sample_size = len(labels)
        refined_labels = np.zeros([sample_size, self._no_labels])
        refined_labels[range(sample_size), labels.astype(int).tolist()] = 1.0
        return refined_labels.astype(float)
    
    def size(self):
        return self._size

    def shuffle(self):
        idx = range(self._size)
        np.random.shuffle(idx)

        self._images = self._images[idx]
        self._labels = self._labels[idx]

    def batches(self, batch_size):
        for start in xrange(0, self._size, batch_size):
            yield (
                self._refine_images(self._images[start : start + batch_size]), 
                self._refine_labels(self._labels[start : start + batch_size]))

    def images(self):
        return self._refine_images(self._images)

    def labels(self):
        return self._refine_labels(self._labels)

