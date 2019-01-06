import os
import tensorflow as tf
import numpy as np
import pandas as pd



class DataGenerator:
    def __init__(self, config):
        self.config = config

        path = self.config.training_data_path
        self.y_raw = pd.read_csv(self.config.labels_file)

        files = [x for x in os.listdir(path) if x[-3:] == 'jpg']
        nfiles = len(files)
        filelist = [[path + x] for x in files]

        print('Converting y data to maps...')
        y_data = self.padder([self.get_y_data(file) for file in files])
        print('Converted')
        filenames = tf.constant(filelist)
        y_data = tf.constant(y_data, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((filenames,y_data))
        dataset = dataset.map(lambda filename, y_arr:self._ondisk_parse_(filename, y_arr)).shuffle(True).batch(self.config.batch_size)

        self.dataset_iterator = dataset.make_one_shot_iterator()

    def _ondisk_parse_(self, filename, y_map):
        '''
        Function applied to every data entry to load the data. The output of this is the input format
        to the model.
        :param filename: filename
        :param y_map: y_data (passes straight through)
        :return:
        '''

        filename_tf = tf.cast(filename[0], tf.string)

        image_string = tf.read_file(filename_tf)
        image = tf.image.decode_jpeg(image_string)
        image = tf.cast(image, tf.float32)

        return dict({'image': image, 'y_map': y_map})

    def get_y_data(self, filename):
        '''
        Converts a filename into an array, with one channel for each boat in the image
        :param filename: filename string
        :return: np array of size [h, w, n_boats_in_this_image]
        '''

        ground_truths = self.y_raw[self.y_raw.ImageId == filename]
        array_of_coords = np.array(ground_truths[['lt_x', 'lt_y', 'rb_x', 'rb_y']])
        # array_of_coords is of shape [n_boxes, 4]

        n_boxes = array_of_coords.shape[0]
        y_map = np.zeros((768, 768, n_boxes))

        for box_idx in range(n_boxes):
            # Loop over amount of boats per image ~ of order 10.
            box = array_of_coords[box_idx, :]
            # TODO: check that im using dimensions correctly here and agrees with static file
            y_map[box[0]:box[2], box[1]:box[3], box_idx] = 1

        return y_map

    def padder(self, list_of_arr):
        '''
        Pads each list of boat maps so all have the same depth (which is the max amount of
        boats across all images) and creates a numpy array of the result.
        :param list_of_arr: list of arrays, each shaped [768, 768, n_boats_in_this_image]
        :return: numpy array of shape [len(list_of_arr), h, w, maximum_n_boats]
        '''

        maximum_n_boats = max([x.shape[2] for x in list_of_arr])
        dim_arr = list_of_arr[0].shape

        self.n_box_max = maximum_n_boats

        b = np.zeros([len(list_of_arr), dim_arr[0], dim_arr[1], maximum_n_boats])
        for i, arr in enumerate(list_of_arr):
            b[i,:, :, :arr.shape[2]] = arr
        return b


