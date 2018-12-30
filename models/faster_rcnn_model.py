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
        self.x: input image
        self.y:  
        '''
        self.x = tf.placeholder(tf.float32, shape=[None, 60, 60, self.config.num_channels])

        # TODO: decide on shape of y
        self.y = tf.placeholder(tf.int64, shape=[None])


        with tf.name_scope('feature_maps'):

            # TODO: extract a feature map. Should return tensor the same shape as the input
            # seperate loss here which has the purpose of feature extraction.

            # Placeholder: identity map as a feature extractor.
            self.feature_maps = tf.identity(self.x, name='identity')

            with tf.name_scope('loss'):
                pass

        with tf.name_scope('region_proposal_network'):

            # TODO: a CNN that slides over the output of feature_maps and classifies and scores boxes
            # loss here is on the box dimensions
            window_size = 3 # TODO: make config
            self.windows = create_windows(self.feature_maps, window_size)
            numb_windows = self.windows.shape[0]

            for window_index in numb_windows:
                window = self.windows[window_index,:,:]

            with tf.name_scope('cnn'):
                flattened_window = create_flatten_layer(window)
                rpn_hidden_layer_size = 128 # TODO: make config
                hidden_layer = create_fc_layer(flattened_window,
                                               num_inputs=tf.shape(flattened_window)[0],
                                               num_outputs=rpn_hidden_layer_size,
                                               use_relu=True)
                k = 9 # Number of proposal boxes TODO: make config
                with tf.name_scope('classification_layer'):
                    self.class_scores = create_fc_layer(hidden_layer,
                                                        num_inputs=rpn_hidden_layer_size,
                                                        num_outputs=2*k,
                                                        use_relu=False)
                with tf.name_scope('regression_layer'):
                    self.reg_scores = create_fc_layer(hidden_layer,
                                                      num_inputs=rpn_hidden_layer_size,
                                                      num_outputs=4*k,
                                                      use_relu=False)



            with tf.name_scope('loss'):
                pass

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=layer_fc2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            self.pred = tf.argmax(layer_fc2, 1)
            correct_prediction = tf.equal(self.pred, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


def create_windows(image, window_shape = 3):
    '''

    :param image: input image
    :param window_size: size of the sliding windows
    :return: [height*width, window_size, window_size] shaped array
    '''

    # Be careful passing big images with small window sizes to this function as
    # (obviously) there is a big blow up in memory
    # (see: http://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.view_as_windows).

    windows = view_as_windows(image, window_shape=window_shape, step=1) # first two dims are height and width
    windows = windows.reshape((-1, window_shape, window_shape))
    return windows