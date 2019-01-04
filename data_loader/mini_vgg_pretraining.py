import os
import pandas as pd
import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.config = config

        # load data here
        pathlist = 'data/train_sample/'
        pathlabels = 'data/train_maps/img_has_boat.csv'

        print('Bottleneck: reading name of all files ...')
        # load file list
        nfiles = len(os.listdir(pathlist))
        # filelist = [os.listdir(pathlist)[i] for i in range(nfiles)]
        filelist = [os.listdir(pathlist)[i] for i in range(10)]
        print('done')
        # load labels
        all_labels = pd.read_csv(pathlabels)

        # find correspondences
        data = all_labels[all_labels.ImageId.isin(filelist)]
        imglist = list(data.ImageId)
        imglist = [pathlist + s for s in imglist]
        labellist = list(data.Img_contains_boat)
        labellist = list(map(int, labellist))

        # translate to tensorflow
        # step 1
        filenames = tf.constant(imglist)
        labels = tf.constant(labellist, dtype=tf.int64)

        # step 2: create a dataset returning slices of `filenames`
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # step 2.1: shuffle
        dataset = dataset.shuffle(buffer_size=len(filelist))

        # step 3: parse every image in the dataset using `map`
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            return {'image': image, 'y_class': label}

        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(self.config.batch_size)

        # step 4: create iterator and final input tensor
        self.iterator = dataset.make_one_shot_iterator()

    def next_batch(self):
        # idx = np.random.choice(500, batch_size)
        # yield self.input[idx], self.y[idx]
        dict_data = self.iterator.get_next()
        yield dict_data
