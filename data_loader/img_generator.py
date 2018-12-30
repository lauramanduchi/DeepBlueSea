import os

import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.config = config

        # load data here
        path = 'data/train_sample/'

        # A vector of filenames.
        nfiles = len(os.listdir(path))
        filelist = [os.listdir(path)[i] for i in range(nfiles)]
        filenames = tf.constant(filelist)

        # `labels[i]` is the label for the image in `filenames[i].
        # TODO: modify the below to read the pickle file with the boxes
        # taken from https://www.tensorflow.org/guide/datasets#decoding_image_data_and_resizing_it
        labels = tf.constant([0, 37, ...])

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(_parse_function)

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label
