import os
import tensorflow as tf


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
        dataset = dataset.map(lambda x:_ondisk_parse_(x)).shuffle(True).batch(batch_size)
        print('Data Loaded')
        self.dataset_iterator = dataset.make_one_shot_iterator()

    def next_batch(self):
        yield self.dataset_iterator.get_next()


def _ondisk_parse_(filename):
    filename = tf.cast(filename, tf.string)

    # TODO: correctly parse label
    label = tf.convert_to_tensor(filename)

    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string)
    image = tf.cast(image, tf.float32)
    return dict({'image': image, 'label': label})
