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
        self.is_training = tf.placeholder(tf.bool)

        with tf.name_scope('data'):

            self.x = tf.placeholder(tf.float32, shape=[None, 768, 768, self.config.num_channels])
            self.y_map = tf.placeholder(tf.float32, shape=[None, 768, 768, None])
            self.y_reg = tf.placeholder(tf.float32, shape=[None, 768, 768, None])

            #data_structure = {'image': tf.float32, 'y_map': tf.float32}
            #data_shape = {'image': tf.TensorShape([None, 768, 768, self.config.num_channels]),
            #             'y_map': tf.TensorShape([None, 768, 768, None])}

            # TODO: add this to the config somehow / move out of code as it only needs to be calculated once
            with tf.name_scope('expand_anchor_shapes_for_reg'):
                anchor_shapes = [(11, 11), (21, 21), (31, 31), (5, 11), (11, 21), (21, 31), (11, 5), (21, 11), (31, 21)]
                y_anchors = np.zeros((768, 768, 1))
                x = np.arange(768)
                X, Y = np.meshgrid(x, x)
                for anchor_shape in anchor_shapes:
                    width = np.ones((768, 768)) * anchor_shape[0]
                    length = np.ones((768, 768)) * anchor_shape[1]
                    y_anchor = np.stack((X, Y, width, length), axis=2)
                    y_anchors = np.concatenate((y_anchors, y_anchor), axis=2)
                y_anchors = y_anchors[:, :, 1:]
                reg_anchors = tf.reshape(tf.convert_to_tensor(y_anchors), shape=[1, 768, 768, 4 * len(anchor_shapes)])
                # this last part needs to stay here with this code construction: we change the first dimension of
                # the reg_anchors to the batch size implicitly (unsure if it works if done explicitly)
                reg_anchors = tf.tile(reg_anchors, multiples=[tf.shape(self.y_reg)[0], 1, 1, 1])
                reg_anchors = tf.cast(reg_anchors, tf.float32)

                # self.handle = tf.placeholder(tf.string, shape=[])
            #iterator = tf.data.Iterator.from_string_handle(self.handle, data_structure, data_shape)

            #next_element = iterator.get_next()

            #self.x = next_element['image']
            #self.y_map = next_element['y_map']

            # TODO: implement y_reg correctly, presently a placeholder
            #self.y_reg = tf.zeros((self.config.batch_size, 768, 768, 4*self.config.n_proposal_boxes))

            with tf.name_scope('expand_y'):
                # Here is where we unwrap the y masks to be IOU maps and then maps that match the loss.
                y_class = []
                y_reg_boat = []
                n_box = tf.shape(self.y_map)[-1]

                for i, anchor_shape in enumerate(anchor_shapes):
                    # Loop over number of anchors ~ of order 9
                    # The y_map is of shape [batch_size, h, w, max_n_boats]
                    # The kenerl we convolve with is of shape [anchor_shape[0], anchor_shape[1], max_n_boats, max_n_boats]
                    # where every entry is 0 except for [:,:, i, i] for all i.
                    # (I think) this is equivalent to running a seperate "all ones" [anchor_shape[0], anchor_shape[1], 1]
                    # kernel over each of the max_n_boats inputs.
                    # TODO: double (triple, quadruple...) check above logic.

                    anchor = tf.zeros((anchor_shape[0], anchor_shape[1], n_box, n_box))
                    diagonal = tf.ones((anchor_shape[0], anchor_shape[1], n_box))
                    anchor_area = anchor_shape[0]*anchor_shape[1]

                    # Assigns ones to anchor[:,:,i,i] (https://www.tensorflow.org/api_docs/python/tf/linalg/set_diag)
                    anchor = tf.linalg.set_diag(anchor, diagonal)

                    # Calculates the intersection of anchor with each gt map in y_map simulatenously (as above)
                    intersection = tf.nn.conv2d(self.y_map, anchor, strides=[1, 1, 1, 1], padding='SAME')

                    # union is the area of the map (per map layer, and per batch entry) + the anchor area (in 2d)
                    # TODO: check that minusing intersection does so entry wise.
                    union = tf.reduce_sum(self.y_map, [1, 2], keepdims=True) + anchor_area - intersection
                    ious = tf.divide(intersection, union)

                    max_iou_over_ground_truth = tf.reduce_max(ious, -1)
                    # for the regression ,we need to know which boat we want to look at
                    argmax_iou_over_ground_truth = tf.argmax(ious, -1)

                    tf.summary.scalar(name='max_gt_iou_' + str(i),
                                      tensor=tf.reduce_max(max_iou_over_ground_truth))

                    summarise_map(name='iou_' + str(i), tensor=max_iou_over_ground_truth)

                    labels = tf.greater(max_iou_over_ground_truth, self.config.iou_threshold)
                    labels = tf.cast(labels, tf.float32)

                    # TODO: test this somehow
                    y_class.append(labels)
                    y_reg_boat.append(argmax_iou_over_ground_truth)
                # Stack all the anchors together in the end this is then of shape [batch, 768, 768, n_anchor]
                self.y_class = tf.stack(y_class, axis=-1)
                y_reg_boat = tf.stack(y_reg_boat, axis=-1)
                if self.config.debug == 1:
                    print('self.y_class.shape', self.y_class.shape)
                    print('y_reg_boat', y_reg_boat.shape)
            tf.summary.image(name='input_images', tensor=self.x, max_outputs=3)
            tf.summary.image(name='y_map', tensor=tf.reduce_sum(self.y_map, -1, keepdims=True), max_outputs=1)


        with tf.name_scope('model'):
            with tf.name_scope('feature_maps'):

                layer_conv1 = create_convolutional_layer(input=self.x,
                                                         num_input_channels=self.config.num_channels,
                                                         conv_filter_size=self.config.filter_size_conv1,
                                                         num_filters=self.config.num_filters_conv1,
                                                         maxpool=0,
                                                         name="conv_layer_1")

                layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                                         num_input_channels=self.config.num_filters_conv1,
                                                         conv_filter_size=self.config.filter_size_conv2,
                                                         num_filters=self.config.num_filters_conv2,
                                                         maxpool=0,
                                                         name='conv_layer_2')

                pool = tf.nn.max_pool(value=layer_conv2,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')

                self.feature_maps = create_deconvolutional_layer(input=pool,
                                                            num_filters=self.config.num_filters_conv2,
                                                            name='deconvolution',
                                                            upscale_factor=2)

                # self.feature_maps = create_convolutional_layer(input=layer_deconv,
                #                                                num_input_channels=self.config.num_filters_conv2,
                #                                                conv_filter_size=self.config.initial_im_size,
                #                                                num_filters=self.config.num_features,
                #                                                maxpool=0)

                tf.summary.image(name='feature_maps', tensor=self.feature_maps[:,:,:,0:3], max_outputs=3)

            with tf.name_scope('region_proposal_network'):

                with tf.name_scope('cnn'):
                    # Realise that the sliding window can be implemented as a convolution
                    with tf.name_scope('sliding_window'):
                        window_outputs = create_convolution(input=self.feature_maps,
                                                            num_input_channels=self.config.num_features,
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

                    # TODO: maybe take mean over batch and sum over other dims
                    classification_loss = tf.reduce_sum(sigmoid_ce)
                    if self.config.debug == 1:
                        print('reg_anchors  ', reg_anchors.shape)
                        print('self.reg_scores ', self.reg_scores.shape)
                        print('self.y_reg  ', self.y_reg.shape)

                    # TODO: this doesnt work
                    # use y_reg_boat to index the right boat
                    # use idx,idy = np.meshgrid(np.arange(768),np.arange(768))
                    # then reg_scores[idx, idy, y_reg_boat]
                    # ... in tensorflow
                    idx, idy = np.meshgrid(np.arange(768), np.arange(768))

                    if self.config.debug == 1:
                        print("checking if n_box is still alive here", n_box)

                    t_x = tf.divide(tf.subtract(self.reg_scores[:, :, :, 0],
                                                reg_anchors[:, :, :, 0]),
                                    reg_anchors[:, :, :, 2])

                    t_x_star = tf.divide(tf.subtract(self.y_reg[:, :, :, 0],
                                                     reg_anchors[:, :, :, 0]),
                                         reg_anchors[:, :, :, 2])
                    t_y = tf.divide(tf.subtract(self.reg_scores[:, :, :, 1],
                                                reg_anchors[:, :, :, 1]),
                                    reg_anchors[:, :, :, 3])
                    t_y_star = tf.divide(tf.subtract(self.y_reg[:, :, :, 1],
                                                     reg_anchors[:, :, :, 1]),
                                         reg_anchors[:, :, :, 3])
                    t_w = tf.log(tf.divide(self.reg_scores[:, :, :, 2],
                                           reg_anchors[:, :, :, 2]))
                    t_w_star = tf.log(tf.divide(self.y_reg[:, :, :, 2],
                                                reg_anchors[:, :, :, 2]))
                    t_h = tf.log(tf.divide(self.reg_scores[:, :, :, 3],
                                           reg_anchors[:, :, :, 3]))
                    t_h_star = tf.log(tf.divide(self.y_reg[:, :, :, 3],
                                                reg_anchors[:, :, :, 3]))
                    if self.config.debug == 1:
                        print(t_x.shape, t_w.shape)
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


