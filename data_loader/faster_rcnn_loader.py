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
        #
        # y_class = []
        # y_reg = []
        #
        # loop = trange(nfiles, desc='', leave=True)
        # for i in loop:
        #     filename = files[i]
        #     loop.set_description(filename)
        #     loop.refresh()
        #
        #     y_data = self.get_y_data(filename)
        #     y_class.append(y_data['y_class'])
        #     y_reg.append(y_data['y_reg'])
        y_data = self.padder([self.get_y_data(file) for file in files])
        filenames = tf.constant(filelist)
        y_data = tf.constant(y_data)

        #
        # y_class = tf.constant(np.array(y_class))
        # y_reg = tf.constant(np.array(y_reg))

        dataset = tf.data.Dataset.from_tensor_slices((filenames,y_data))
        dataset = dataset.map(lambda filename, y_arr:self._ondisk_parse_(filename, y_arr)).shuffle(True).batch(self.config.batch_size)

        self.dataset_iterator = dataset.make_one_shot_iterator()


    def _ondisk_parse_(self, filename, y_map):

        filename_tf = tf.cast(filename[0], tf.string)

        image_string = tf.read_file(filename_tf)
        image = tf.image.decode_jpeg(image_string)
        image = tf.cast(image, tf.float32)

        anchor_shapes = [(11, 11), (21, 21), (31, 31), (5, 11), (11, 21), (21, 31), (11, 5), (21, 11), (31, 21)]
        y_class = []
        y_reg = tf.zeros((768, 768, 4 * self.config.n_proposal_boxes))

        for i, anchor_shape in enumerate(anchor_shapes):
            # Loop over number of anchors ~ of order 9
            y_class.append(self.calculate_ious_tf(y_map, anchor_shape))

        y_class = tf.stack(y_class, axis=-1)

        return dict({'image': image, 'y_class': y_class, 'y_reg': y_reg})

    def get_y_data(self, filename):

        ground_truths = self.y_raw[self.y_raw.ImageId == filename]
        array_of_coords = np.array(ground_truths[['lt_x', 'lt_y', 'rb_x', 'rb_y']])
        # array_of_coords is of shape [n_boxes, 4]

        n_boxes = array_of_coords.shape[0]
        y_map = np.zeros((768, 768, n_boxes))

        for box_idx in range(n_boxes):
            # Loop over amount of boats per image ~ of order 10.
            box = array_of_coords[box_idx, :]
            y_map[box[0]:box[2], box[1]:box[3], box_idx] = 1

        # anchor_shapes = [(11,11), (21,21), (31,31), (5,11), (11,21), (21,31), (11,5), (21,11), (31,21)]
        # y_class = np.zeros((768, 768, self.config.n_proposal_boxes))
        #
        # if array_of_coords.shape[0] != 0:
        #     for i,anchor_shape in enumerate(anchor_shapes):
        #         # Loop over number of anchors ~ of order 9
        #         y_class[:, :, i] = self.calculate_ious(array_of_coords, anchor_shape)
        #
        # y_reg = np.zeros((768, 768, 4*self.config.n_proposal_boxes))

        return y_map

    def padder(self, list_of_arr):

        # list of arr is list of arr shaped [768, 768, n_boxes]

        maximum_n_boats = max([x.shape[2] for x in list_of_arr])
        dim_arr = list_of_arr[0].shape

        self.n_box_max = maximum_n_boats

        b = np.zeros([len(list_of_arr), dim_arr[0], dim_arr[1], maximum_n_boats])
        for i, arr in enumerate(list_of_arr):
            b[i,:, :, :arr.shape[0]] = arr
        return b

    def calculate_ious_tf(self, y_map, anchor_shape):

        # TODO: make this faster with one covolution over all boxes
        n_box = self.n_box_max

        anchor = np.zeros((anchor_shape[0],anchor_shape[1], n_box, n_box))
        for i in range(anchor.shape[2]):
            anchor[:,:,i,i] = 1

        anchor = tf.constant(anchor)

        ious = self.iou_matrix_tf(y_map, anchor)

        max_iou_over_gt = tf.math.reduce_max(ious, 2)
        thresholded_iou = tf.math.greater(max_iou_over_gt, self.config.iou_threshold)

        return tf.cast(thresholded_iou,  dtype=tf.int32)

    def iou_matrix_tf(self, box_map, anchor):
        box_map = tf.expand_dims(box_map, 0)

        intersection = tf.nn.conv2d(box_map, anchor, strides=[1, 1, 1, 1], padding='SAME')
        # TODO: make union per layer
        union = tf.math.reduce_sum(anchor) + tf.math.reduce_sum(box_map)
        iou = tf.math.divide(intersection, union)

        return tf.squeeze(iou)


