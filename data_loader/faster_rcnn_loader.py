import os
import tensorflow as tf
import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, config):
        self.config = config

        path = self.config.training_data_path

        nfiles = len(os.listdir(path))
        filelist = [path + os.listdir(path)[i] for i in range(nfiles)]
        filenames = tf.constant(filelist)

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(lambda x:self._ondisk_parse_(x)).shuffle(True).batch(self.config.batch_size)

        self.dataset_iterator = dataset.make_one_shot_iterator()


    def _ondisk_parse_(self, filename):

        filename = tf.cast(filename, tf.string)

        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string)
        image = tf.cast(image, tf.float32)

        # TODO: replace this with an actual loading in of the matrix

        y_class = np.zeros((768, 768, 2*self.config.n_proposal_boxes))
        y_reg = np.zeros((768, 768, 4*self.config.n_proposal_boxes))

        # TODO: Generate the ones in the matrix on the fly from the csv.
        y_class = tf.convert_to_tensor(y_class, dtype=tf.int16)
        y_reg = tf.convert_to_tensor(y_reg, dtype=tf.int16)

        return dict({'image': image, 'y_class': y_class, 'y_reg': y_reg})
