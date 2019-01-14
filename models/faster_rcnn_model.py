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
            self.y_reg = tf.placeholder(tf.float32, shape=[None, 768, 768, None, 4])

            #data_structure = {'image': tf.float32, 'y_map': tf.float32}
            #data_shape = {'image': tf.TensorShape([None, 768, 768, self.config.num_channels]),
            #             'y_map': tf.TensorShape([None, 768, 768, None])}

            # TODO: add this to the config somehow / move out of code as it only needs to be calculated once
            with tf.name_scope('expand_anchor_shapes_for_reg'):
                anchor_shapes = [(21,21), (21, 41), (41,21), (41, 81), (81, 41), (51,51), (151,81), (81, 151), (101,101), (201,201)]
                n_anchors = len(anchor_shapes)
                y_anchors = np.zeros((1, 768, 768, n_anchors, 4))
                x = np.arange(768)
                X, Y = np.meshgrid(x, x)
                y_anchors[:, :, :, :, 0] = np.reshape(Y, (1, 768, 768, 1))
                y_anchors[:, :, :, :, 1] = np.reshape(X, (1, 768, 768, 1))
                i = 0
                for anchor_shape in anchor_shapes:
                    y_anchors[:, :, :, i, 2] = np.ones((1, 768, 768)) * anchor_shape[0]  # width
                    y_anchors[:, :, :, i, 3] = np.ones((1, 768, 768)) * anchor_shape[1]  # length
                    i += 1
                # this last part needs to stay here with this code construction: we change the first dimension of
                # the reg_anchors to the batch size implicitly (unsure if it works if done explicitly)
                reg_anchors = tf.tile(y_anchors, multiples=[tf.shape(self.y_reg)[0], 1, 1, 1, 1])
                reg_anchors = tf.cast(reg_anchors, tf.float32)

                if self.config.debug == 1:
                    print("reg_anchors", reg_anchors.shape)
                # self.handle = tf.placeholder(tf.string, shape=[])
            #iterator = tf.data.Iterator.from_string_handle(self.handle, data_structure, data_shape)

            #next_element = iterator.get_next()

            #self.x = next_element['image']
            #self.y_map = next_element['y_map']

            with tf.name_scope('expand_y'):
                # Here is where we unwrap the y masks to be IOU maps and then maps that match the loss.
                y_class = []
                selected_boat_index = []
                iou_mask = []
                iou_average_for_summary = []
                pos_mask = []
                neg_mask = []
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

                    iou_average_for_summary.append(max_iou_over_ground_truth)

                    labels = tf.greater(max_iou_over_ground_truth, self.config.iou_positive_threshold)
                    labels = tf.cast(labels, tf.float32)

                    # We only deal in both losses with boxes with IOU above a upper threshold and
                    # below a lower threshold and so here we create a mask which will be 1 for
                    # all such iou scores and 0 for those inside the threshold so we can
                    # use it as a weighting for the losses
                    pos_labels = tf.cast(tf.greater(max_iou_over_ground_truth,
                                                    self.config.iou_positive_threshold), tf.float32)
                    neg_labels = tf.cast(tf.less(max_iou_over_ground_truth,
                                                 self.config.iou_negative_threshold), tf.float32)
                    if self.config.debug:
                        print("pos_labels", pos_labels.shape)


                    pos_mask.append(pos_labels)
                    neg_mask.append(neg_labels)

                    #iou_mask_anchor = pos_labels + neg_labels
                    # iou_mask shape: [batch, 768, 768, n_proposal_boxes]
                    #iou_mask.append(tf.cast(iou_mask_anchor, tf.float32))


                    y_class.append(labels)
                    selected_boat_index.append(argmax_iou_over_ground_truth)

                # Stack all the anchors together in the end this is then of shape [batch, 768, 768, n_anchor]
                self.y_class = tf.stack(y_class, axis=-1)
                selected_boat_index = tf.stack(selected_boat_index, axis=-1)

                # Stack IOU masks
                temp_pos_mask = tf.stack(pos_mask, -1)
                temp_neg_mask = tf.stack(neg_mask, -1)
                #iou_mask = tf.stack(iou_mask, -1)

                if self.config.debug == 1:
                    print('self.y_class.shape', self.y_class.shape)
                    print('selected_boat_index', selected_boat_index.shape)
                    print('temp_pos_mask', temp_pos_mask.shape)

                # Stack iou_average_for_summary
                iou_average_for_summary = tf.stack(iou_average_for_summary, -1)
                iou_average_for_summary = tf.reduce_mean(iou_average_for_summary, -1)
                tf.summary.scalar(name='max_over_pixels_average_groundtruth_iou_over_anchors',
                                  tensor=tf.reduce_max(iou_average_for_summary))

                summarise_map(name='average_groundtruth_iou_over_anchors', tensor=iou_average_for_summary)

                # Filter y_reg by which boxes are positive
                y_reg_gt = []
                with tf.name_scope('filter_groundtruth_regression'):
                    for k in range(n_anchors):
                        # For each anchor run select_with_matrix_tf which filters the map of regression
                        # coordinates y_reg to the (per pixel) particular boat indicated by selected_boat_index
                        y_reg_gt_anchor = select_with_matrix_tf(tensor=self.y_reg,
                                                                indexer=selected_boat_index[:,:,:, k],
                                                                batch_size=self.config.batch_size)
                        y_reg_gt.append(y_reg_gt_anchor)

                    self.y_reg_gt = tf.stack(y_reg_gt, -2)

            # Some summaries
            tf.summary.image(name='input_images', tensor=self.x, max_outputs=3)
            tf.summary.image(name='y_map', tensor=tf.reduce_sum(self.y_map, -1, keepdims=True), max_outputs=1)

            with tf.name_scope('sample'):
                # counting the number of positive samples.
                # if there are zero, then  we are in a "no_boats" batch image and sample consequently
                # note: instead of tf.reduce_sum(self.y_map) > 0, we want tf.reduce_sum(self.y_map, [1,2,3])
                # the latter gives us the sum of the ground truth per image
                # if the sum is positive, then there are boats in the image and we want to sample some
                # TODO: fix :)
                class_mask = []
                pos_mask = []

                for i in range(self.config.batch_size):
                    n_positive_samples = tf.cond(tf.reduce_sum(temp_pos_mask[i]) > 0,
                                             lambda: self.config.n_positive_samples,
                                             lambda: 0)
                    n_negative_samples = tf.cond(tf.reduce_sum(temp_pos_mask[i]) > 0,
                                             lambda: self.config.n_negative_samples,
                                             lambda: self.config.n_negative_samples_when_no_boats)
                    # sampling
                    # note that atm pos_sample applies the same sampling over the whole batch
                    # refer to utils for the function
                    # TODO: fix :)
                    pos_sample = tf.py_func(np_sample, [temp_pos_mask[i], 1, n_positive_samples], tf.float64)
                    pos_sample = tf.cast(pos_sample, tf.float32)
                    #pos_mask = temp_pos_mask * pos_sample
                    neg_sample = tf.py_func(np_sample, [temp_neg_mask[i], 1, n_negative_samples], tf.float64)
                    neg_sample = tf.cast(neg_sample, tf.float32)
                    #neg_mask = temp_neg_mask * neg_sample

                    if self.config.debug:
                       print("pos_sample.shape", pos_sample.shape)
                       print("temp_pos_mask.shape", temp_pos_mask.shape)
                    temp_class_mask = pos_sample + neg_sample
                    class_mask.append(temp_class_mask)
                    pos_mask.append(pos_sample)

                class_mask = tf.stack(class_mask, 0)
                pos_mask = tf.stack(pos_mask, 0)



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
                        # reg_outputs: [batch_size, 768, 768, 4*self.config.n_proposal_boxes]
                        reg_outputs = create_convolution(input=window_outputs,
                                                             num_input_channels=self.config.sliding_hidden_layer_size,
                                                             conv_filter_size=1,
                                                             num_filters=4*self.config.n_proposal_boxes,
                                                             stride=1,
                                                             data_format="NHWC")

                        # self.reg_scores: [batch_size, 768, 768, self.config.n_proposal_boxes, 4]
                        self.reg_scores = tf.reshape(tensor=reg_outputs,
                                                     shape=[-1, 768, 768, self.config.n_proposal_boxes, 4],
                                                     name='reshape_regression_outputs')



            with tf.name_scope('loss'):

                sigmoid_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_class,
                                                                     logits=self.class_scores)
                # Remember that we only look at positive (> upper iou thresh) and negative (< iou thresh) boxes
                print('IOU Mask Shape', class_mask.shape)
                masked_signoid_ce = tf.multiply(class_mask, sigmoid_ce)

                classification_loss_per_sample = tf.reduce_sum(masked_signoid_ce, axis = [1,2,3])
                classification_loss = tf.reduce_mean(classification_loss_per_sample)

                if self.config.debug == 1:
                    print('sigmoid_ce  ', sigmoid_ce.shape)
                    print('classification_loss  ', classification_loss.shape)


                with tf.name_scope('create_adjusted_coordinates'):
                    # t_x = (x_predict - x_anchor)/ w_anchor
                    t_x = (self.reg_scores[:, :, :, :, 0] - reg_anchors[:, :, :, :, 0]) / reg_anchors[:, :, :, :, 2]
                    t_x_star = (self.y_reg_gt[:, :, :, :, 0] - reg_anchors[:, :, :, :, 0]) / reg_anchors[:, :, :, :, 2]
                    # t_y = (y_predict - y_anchor)/ h_anchor
                    t_y = (self.reg_scores[:, :, :, :, 1] - reg_anchors[:, :, :, :, 1]) / reg_anchors[:, :, :, :, 3]
                    t_y_star = (self.y_reg_gt[:, :, :, :, 1] - reg_anchors[:, :, :, :, 1]) / reg_anchors[:, :, :, :, 3]
                    # t_w = log(w_predict / w_anchor)
                    t_w_star = tf.log(self.y_reg_gt[:, :, :, :, 2] / reg_anchors[:, :, :, :, 2])
                    t_w = tf.maximum(tf.log(self.reg_scores[:, :, :, :, 2] / reg_anchors[:, :, :, :, 2]),
                                     tf.cast(tf.fill(dims=tf.shape(t_w_star), value=-100000), dtype=tf.float32))
                    # tf.constant(-100000, shape=tf.shape(t_w_star)))
                    # t_h = log(h_predict / h_anchor)
                    t_h_star = tf.log(self.y_reg_gt[:, :, :, :, 3] / reg_anchors[:, :, :, :, 3])
                    t_h = tf.maximum(tf.log(self.reg_scores[:, :, :, :, 3] / reg_anchors[:, :, :, :, 3]),
                                     tf.cast(tf.fill(dims=tf.shape(t_w_star), value=-100000), dtype=tf.float32))
                    #tf.constant(-100000, shape=tf.shape(t_x)))
                    y_reg_loss_pred = tf.stack([t_x, t_y, t_w, t_h], axis=4)
                    y_reg_loss_gt = tf.stack([t_x_star, t_y_star, t_w_star, t_h_star], axis=4)


                regression_loss_per_pixel = tf.losses.mean_squared_error(labels=y_reg_loss_gt,
                                                                         predictions=y_reg_loss_pred,
                                                                         reduction=tf.losses.Reduction.NONE)

                # Remember that we only look at positive (> upper iou thresh) boxes for regression
                # Expand mask to have dimension for 4 coordiantes. Now of shape [batch, 768, 768, n_proposal_boxes, 4]
                iou_mask_regression = tf.tile(tf.expand_dims(self.y_class, -1), [1,1,1,1,4])

                # Remember that we only look at positive (> upper iou thresh) boxes
                # Expand mask to have dimension for 4 coordiantes. Now of shape [batch, 768, 768, n_proposal_boxes, 4]
                iou_mask_regression = tf.tile(tf.expand_dims(pos_mask, -1), [1, 1, 1, 1, 4])

                masked_regression_loss_per_pixel = tf.multiply(regression_loss_per_pixel, iou_mask_regression)

                # For summary, summarise the regression loss for each of the 4 coordinates seperately to see
                # if any particular one is under peforming.
                seperate_coordinate_losses = tf.reduce_mean(
                    tf.reduce_sum(masked_regression_loss_per_pixel, axis=[1,2,3]),
                    axis=0
                )

                tf.summary.scalar(name='x_center_reg_loss', tensor=seperate_coordinate_losses[0])
                tf.summary.scalar(name='y_center_reg_loss', tensor=seperate_coordinate_losses[1])
                tf.summary.scalar(name='width_reg_loss', tensor=seperate_coordinate_losses[2])
                tf.summary.scalar(name='height_reg_loss', tensor=seperate_coordinate_losses[3])


                regression_loss_per_sample = tf.reduce_sum(masked_regression_loss_per_pixel, axis=[1,2,3,4])
                regression_loss = tf.reduce_mean(regression_loss_per_sample)

                self.loss = classification_loss + regression_loss


        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)

        with tf.name_scope('summaries'):
            self.summaries = tf.summary.merge_all()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


