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
        self.y_raw = pd.read_csv(self.config.labels_file)


    def _ondisk_parse_(self, filename):

        filename_tf = tf.cast(filename, tf.string)

        image_string = tf.read_file(filename_tf)
        image = tf.image.decode_jpeg(image_string)
        image = tf.cast(image, tf.float32)

        # TODO: replace this with an actual loading in of the matrix

        y_class, y_reg = self.classification_data_parser(filename, n_proposals=self.config.n_proposal_boxes)

        # TODO: Generate the ones in the matrix on the fly from the csv.
        y_class = tf.convert_to_tensor(y_class, dtype=tf.int16)
        y_reg = tf.convert_to_tensor(y_reg, dtype=tf.int16)

        return dict({'image': image, 'y_class': y_class, 'y_reg': y_reg})

    def classification_data_parser(self, filename, n_proposals, iou_thresh=0.7):
        # TODO: include the widths and ratios of anchors in a cleaner format
        # Note this formualtion of anchor dimension is temporary
        anchor_widths = [9, 21, 41, 5, 9, 21, 9, 9, 41]
        anchor_heights = [9, 21, 41, 9, 21, 41, 5, 9, 41]

        # Select rows from csv that contain the neccesary images
        ground_truths = self.y_raw[self.y_raw.ImageId == filename]
        y_class = np.zeros((768, 768, 2 * self.config.n_proposal_boxes))
        y_reg = np.zeros((768, 768, 4 * self.config.n_proposal_boxes))

        if ground_truths.empty:
            return y_class, y_reg
        else:
            # iterate over every pixel running per_pixel_parse
            # TODO: write this more elegantly
            for i in range(768):
                for j in range(768):
                    pixel_coord = (i, j) # TODO: ensure correct pixel ordering
                    y_class[j, i, :] = self.per_pixel_parse(pixel_coord = pixel_coord,
                                                            anchor_widths=anchor_widths,
                                                            anchor_heights=anchor_heights,
                                                            ground_truths=ground_truths,
                                                            iou_thresh=self.config.iou_threshold)





    def per_pixel_parse(self, pixel_coord, anchor_widths, anchor_heights, ground_truths, iou_thresh):
        '''

        :param pixel_coord: i, j coordinate of pixel where i is x and j is y. FEED AS i,j NOT j,i LIKE NUMPY
        :param anchor_widths:
        :param anchor_heights:
        :param ground_truths:
        :param iou_thresh:
        :return:
        '''
        ious = []
        for anchor_width, anchor_height in zip(anchor_widths, anchor_heights):
            across_gt_ious = []
            for _, gt_box in ground_truths.iterrows():
                anchor_half_w, anchor_half_h = (anchor_width-1)/2, (anchor_height-1)/2

                anchor_left_bottom = (pixel_coord[0] - anchor_half_w, pixel_coord[1] - anchor_half_h)
                anchor_right_top = (pixel_coord[0] + anchor_half_w, pixel_coord[1] + anchor_half_h)
                across_gt_ious.append(iou(gt_box['lb'], gt_box['rt'], anchor_left_bottom, anchor_right_top))
            ious.append(max(across_gt_ious))

        return (np.array(ious) >= iou_thresh).astype(int)


def iou(left_bottom_1, right_top_1, left_bottom_2, right_top_2):

    itr_lr = (max(left_bottom_1[0], left_bottom_2[0]), max(left_bottom_1[1], left_bottom_2[1]))
    itr_rt = (min(right_top_1[0], right_top_2[0]), min(right_top_1[1], right_top_2[1]))

    itr_area = area(itr_lr, itr_rt)
    uni_area = area(left_bottom_1, right_top_1) + area(left_bottom_2, right_top_2) - itr_area
    return itr_area / uni_area


def area(lb, rt):
    h = max(0, rt[1] - lb[1])
    w = max(0, rt[0] - lb[0])

    return h * w
