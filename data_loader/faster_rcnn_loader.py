import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import image as mpimg



class DataGenerator:
    def __init__(self, config):
        self.config = config

        path = self.config.training_data_path
        self.y_raw = pd.read_csv(self.config.labels_file)

        files = [x for x in os.listdir(path) if x[-3:] == 'jpg']
        if config.debug == 1:
            files = files[0:2]
        #filelist = [[path + x] for x in files[0:2]] # remember to delete [0:2] !!

        #files = np.asarray(files)

        #print('Converting y data to maps...')
        #y_data = self.padder([self.get_y_data(file) for file in files[0:2]])
        #print('Converted')

        #filenames = tf.constant(filelist)
        #y_data = tf.constant(y_data, dtype=tf.float32)

        #dataset = tf.data.Dataset.from_tensor_slices((filenames,y_data))
        #dataset = dataset.map(lambda filename, y_arr:self._ondisk_parse_(filename, y_arr)).shuffle(True).batch(self.config.batch_size)

        #self.dataset_iterator = dataset.make_one_shot_iterator()

        self.input, self.input_dev = train_test_split(files, test_size=self.config.val_split)

    def next_batch(self, batch_size):
        idx = np.random.choice(len(self.input), batch_size)
        sub_input = [self.input[i] for i in idx]
        filenames = [self.config.training_data_path + x for x in sub_input]
        input = self.read_images(filenames)

        y_data = []
        y_reg = []
        for file in sub_input:
            out_y_data = self.get_y_data(file)
            y_data.append(out_y_data[0])
            y_reg.append(out_y_data[1])

        y_data = self.padder(y_data)
        y_reg = self.padder_coord(y_reg)

        yield input, y_data, y_reg

    def next_batch_dev(self, batch_size):
        idx = np.random.choice(len(self.input_dev), batch_size)
        sub_input_dev = [self.input_dev[i] for i in idx]
        filenames_dev = [self.config.training_data_path + x for x in sub_input_dev]
        input_dev = self.read_images(filenames_dev)

        y_data_dev = []
        y_reg_dev = []
        for file in sub_input_dev:
            out_y_data = self.get_y_data(file)
            y_data_dev.append(out_y_data[0])
            y_reg_dev.append(out_y_data[1])
        y_data_dev = self.padder(y_data_dev)
        y_reg_dev = self.padder_coord(y_reg_dev)

        yield input_dev, y_data_dev, y_reg_dev

    def read_images(self, filenames):
        '''
        Function applied to every data entry to load the data. The output of this is the input format
        to the model.
        :param filename: filename
        :param y_map: y_data (passes straight through)
        :return:
        '''

        imgs = []
        for i in range(len(filenames)):
            print(filenames[i])
            imgs.append(mpimg.imread(filenames[i]))
        imgs = np.asarray(imgs)
        return imgs

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
        y_reg = np.zeros((768, 768, n_boxes, 4))

        for box_idx in range(n_boxes):
            # Loop over amount of boats per image ~ of order 10.
            box = array_of_coords[box_idx, :]
            y_map[box[0]:box[2], box[1]:box[3], box_idx] = 1

            # Add boat box coordinates to y_reg
            y_reg[:, :, box_idx, 0] = round((box[2] + box[0]) / 2)  # x-centre
            y_reg[:, :, box_idx, 1] = round((box[3] + box[1]) / 2)  # y-centre
            y_reg[:, :, box_idx, 2] = box[2] - box[0]  # width
            y_reg[:, :, box_idx, 3] = box[3] - box[1]  # height

        return y_map, y_reg

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

    def padder_coord(self, list_of_arr):
        '''
        Pads each list of boat maps so all have the same depth (which is the max amount of
        boats across all images) and creates a numpy array of the result.
        :param list_of_arr: list of arrays, each shaped [768, 768, n_boats_in_this_image, depth] (depth typically 4)
        :return: numpy array of shape [len(list_of_arr), h, w, maximum_n_boats, depth]
        '''

        maximum_n_boats = max([x.shape[2] for x in list_of_arr])
        dim_arr = list_of_arr[0].shape
        depth = dim_arr[-1]

        b = np.zeros([len(list_of_arr), dim_arr[0], dim_arr[1], maximum_n_boats, depth])
        for i, arr in enumerate(list_of_arr):
            b[i, :, :, :arr.shape[2], :] = arr
        return b
