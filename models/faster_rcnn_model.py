from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
from models.utils_model import *

# STATUS: At least appears to train but needs verification, regression loading and loss not
# implemented.


class FasterRcnnModel(BaseModel):
    '''
    Model for the entire (adapted) Faster RCNN we use here
    '''

    def __init__(self, config):
        super(FasterRcnnModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # self.is_training = tf.placeholder(tf.bool)

        '''
        self.x: input image batch [batch_size, 768, 768, num_channels]
        self.y:  
        '''
        with tf.name_scope('data'):

            data_structure = {'image': tf.float32, 'y_map': tf.float32}
            data_shape = {'image': tf.TensorShape([None, 768, 768, self.config.num_channels]),
                          'y_map': tf.TensorShape([None, 768, 768, None])}

            # TODO: add this to the config somehow
            anchor_shapes = [(11, 11), (21, 21), (31, 31), (5, 11), (11, 21), (21, 31), (11, 5), (21, 11), (31, 21)]

            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(self.handle, data_structure, data_shape)

            next_element = iterator.get_next()

            self.x = next_element['image']
            self.y_map = next_element['y_map']

            # TODO: implement y_reg correctly, presently a placeholder
            self.y_reg = tf.zeros((self.config.batch_size, 768, 768, 4*self.config.n_proposal_boxes))

            with tf.name_scope('expand_y'):
                # Here is where we unwrap the y masks to be IOU maps and then maps that match the loss.
                y_class = []

                for i, anchor_shape in enumerate(anchor_shapes):
                    # Loop over number of anchors ~ of order 9
                    # The y_map is of shape [batch_size, h, w, max_n_boats]
                    # The kenerl we convolve with is of shape [anchor_shape[0], anchor_shape[1], max_n_boats, max_n_boats]
                    # where every entry is 0 except for [:,:, i, i] for all i.
                    # (I think) this is equivalent to running a seperate "all ones" [anchor_shape[0], anchor_shape[1], 1]
                    # kernel over each of the max_n_boats inputs.
                    # TODO: double (triple, quadruple...) check above logic.

                    n_box = tf.shape(self.y_map)[-1]
                    anchor = tf.zeros((anchor_shape[0], anchor_shape[1], n_box, n_box))
                    diagonal = tf.ones((anchor_shape[0], anchor_shape[1], n_box))
                    anchor_area = anchor_shape[0]*anchor_shape[1]

                    # Assigns ones to anchor[:,:,i,i] (https://www.tensorflow.org/api_docs/python/tf/linalg/set_diag)
                    anchor = tf.linalg.set_diag(anchor, diagonal)

                    # Calculates the intersection of anchor with each gt map in y_map simulatenously (as above)
                    intersection = tf.nn.conv2d(self.y_map, anchor, strides=[1, 1, 1, 1], padding='SAME')

                    # union is the area of the map (per map layer, and per batch entry) + the anchor area (in 2d)
                    union = tf.reduce_sum(self.y_map, [1, 2], keepdims=True) + anchor_area
                    ious = tf.divide(intersection, union)
                    max_iou_over_ground_truth = tf.reduce_max(ious, -1)
                    labels = tf.greater(max_iou_over_ground_truth, self.config.iou_threshold)
                    labels = tf.cast(labels, tf.float32)
                    # TODO: test this somehow

                    y_class.append(labels)

                # Stack all the anchors together in the end this is then of shape [batch, 768, 768, n_anchor]
                self.y_class = tf.stack(y_class, axis=-1)

            tf.summary.image(name='input_images', tensor=self.x, max_outputs=3)

            visualise_y_map = tf.expand_dims(self.y_map[:,:,:,0], -1)
            tf.summary.image(name='y_map', tensor=visualise_y_map, max_outputs=1)

        with tf.name_scope('model'):
            with tf.name_scope('feature_maps'):

                # TODO: extract a feature map. Should return tensor the same shape as the input
                # seperate loss here which has the purpose of feature extraction.

                # Placeholder: identity map as a feature extractor.
                self.feature_maps = tf.identity(self.x, name='identity')

                tf.summary.image(name='feature_maps', tensor=self.feature_maps, max_outputs=1)

                with tf.name_scope('loss'):
                    pass

            with tf.name_scope('region_proposal_network'):

                with tf.name_scope('cnn'):
                    # Realise that the sliding window can be implemented as a convolution
                    with tf.name_scope('sliding_window'):
                        window_outputs = create_convolution(input=self.feature_maps,
                                                            num_input_channels=self.config.num_channels,
                                                            conv_filter_size=self.config.window_size,
                                                            num_filters=self.config.sliding_hidden_layer_size,
                                                            stride=1,
                                                            data_format="NHWC")

                    with tf.name_scope('classification_layer'):
                        self.class_scores = create_convolution(input=window_outputs,
                                                               num_input_channels=self.config.sliding_hidden_layer_size,
                                                               conv_filter_size=1,
                                                               num_filters=self.config.n_proposal_boxes,
                                                               stride=1,
                                                               data_format="NHWC")
                    with tf.name_scope('regression_layer'):
                        self.reg_scores = create_convolution(input=window_outputs,
                                                             num_input_channels=self.config.sliding_hidden_layer_size,
                                                             conv_filter_size=1,
                                                             num_filters=4*self.config.n_proposal_boxes,
                                                             stride=1,
                                                             data_format="NHWC")



                with tf.name_scope('loss'):

                    sigmoid_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_class,
                                                                         logits=self.class_scores)

                    classification_loss = tf.reduce_sum(sigmoid_ce)
                    regression_loss = 0 #tf.losses.mean_squared_error(labels=self.y_reg, predictions=self.reg_scores)

                    self.loss = classification_loss + regression_loss


                    # TODO: build correct loss
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)

        with tf.name_scope('summaries'):
            self.summaries = tf.summary.merge_all()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


