from base.base_model import BaseModel
import tensorflow as tf
from models.utils_model import *
from skimage.util.shape import view_as_windows

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
        self.is_training = tf.placeholder(tf.bool)

        '''
        self.x: input image batch [batch_size, 768, 768, num_channels]
        self.y:  
        '''
        self.x = tf.placeholder(tf.float32, shape=[None, 768, 768, self.config.num_channels])

        self.class_y = tf.placeholder(tf.float32, shape=[None, 768, 768, 2*self.config.n_proposal_boxes])
        self.reg_y = tf.placeholder(tf.float32, shape=[None, 768, 768, 2 * self.config.n_proposal_boxes])


        with tf.name_scope('feature_maps'):

            # TODO: extract a feature map. Should return tensor the same shape as the input
            # seperate loss here which has the purpose of feature extraction.

            # Placeholder: identity map as a feature extractor.
            self.feature_maps = tf.identity(self.x, name='identity')

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
                # TODO: add a loss
                pass


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


