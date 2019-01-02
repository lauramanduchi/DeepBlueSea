from base.base_model import BaseModel
import tensorflow as tf
from models.utils_model import *

# NOTE: WIP, PROBABLY DOESNT WORK YET


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
            data_structure = {'image': tf.float32, 'y_class': tf.int16, 'y_reg': tf.int16}
            data_shape = {'image': tf.TensorShape([None, 768, 768, self.config.num_channels]),
                          'y_class': tf.TensorShape([None, 768, 768, 2*self.config.n_proposal_boxes]),
                          'y_reg': tf.TensorShape([None, 768, 768, 4*self.config.n_proposal_boxes])}

            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(self.handle, data_structure, data_shape)

            next_element = iterator.get_next()

            #self.x = tf.placeholder(tf.float32, shape=[None, 768, 768, self.config.num_channels])

            #self.class_y = tf.placeholder(tf.float32, shape=[None, 768, 768, 2*self.config.n_proposal_boxes])
            #self.reg_y = tf.placeholder(tf.float32, shape=[None, 768, 768, 4*self.config.n_proposal_boxes])

            self.x = next_element['image']
            self.class_y = next_element['y_class']
            self.reg_y = next_element['y_reg']

            tf.summary.image(name = 'input_images', tensor=self.x, max_outputs=3)

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
                                                               num_filters=2*self.config.n_proposal_boxes,
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

                    #self.cross_entropy =
                    self.mse = tf.losses.mean_squared_error(labels=self.reg_y,
                                                            predictions=self.reg_scores)


                    # TODO: build correct loss
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.mse,
                                                                                         global_step=self.global_step_tensor)

        with tf.name_scope('summaries'):
            self.summaries = tf.summary.merge_all()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


