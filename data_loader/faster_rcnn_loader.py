import os
import tensorflow as tf
import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self, config):
        self.config = config

        # TODO: add path and batch size to config
        path = '../data/test_mini/'
        batch_size = 64

        nfiles = len(os.listdir(path))
        filelist = [path + os.listdir(path)[i] for i in range(nfiles)]
        filenames = tf.constant(filelist)

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        print('Loading Data')
        dataset = dataset.map(lambda x:self._ondisk_parse_(x)).shuffle(True).batch(batch_size)
        print('Data Loaded')
        self.dataset_iterator = dataset.make_one_shot_iterator()

        # Load the labels csv
        # TODO: pandas from csv

    def next_batch(self):
        yield self.dataset_iterator.get_next()


    def _ondisk_parse_(self, filename):
        filename = tf.cast(filename, tf.string)

        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string)
        image = tf.cast(image, tf.float32)

        # TODO: replace this with an actual loading in of the matrix

        y_class = np.zeros((768, 768, 2*self.config.n_proposal_boxes))
        y_reg = np.zeros((768, 768, 4*self.config.n_proposal_boxes))

        # Here: Generate the ones in the matrix on the fly from the csv.

        return dict({'image': image, 'y_class': y_class, 'y_reg': y_reg})
