from base.base_model import BaseModel
from models.utils_model import *


class MiniVGG(BaseModel):
    def __init__(self, config):
        super(MiniVGG, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # below no longer needed
        """
        self.x = tf.placeholder(tf.float32, shape=[None,
                                                   self.config.initial_im_size,
                                                   self.config.initial_im_size,
                                                   self.config.num_channels])

        self.y = tf.placeholder(tf.int64, shape=[None])
        """

        data_structure = {'image': tf.float32, 'y_class': tf.int64}
        data_shape = {'image': tf.TensorShape([None,
                                               self.config.initial_im_size,
                                               self.config.initial_im_size,
                                               self.config.num_channels]),
                      'y_class': tf.TensorShape([None])}

        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.handle, data_structure, data_shape)

        next_element = iterator.get_next()

        self.x = next_element['image']
        self.y = next_element['y_class']

        tf.summary.image(name='input_images', tensor=self.x, max_outputs=3)
        tf.summary.image(name='y_class', tensor=tf.cast(self.y, tf.uint8), max_outputs=3)

        # network architecture
        # two convolutional layers with ReLU activation
        # one maxpool layer
        # one convolutional layer with final size equal to image size
        # one fully convolutional layer for classification

        layer_conv1 = create_convolutional_layer(input=self.x,
                                                 num_input_channels=self.config.num_channels,
                                                 conv_filter_size=self.config.filter_size_conv1,
                                                 num_filters=self.config.num_filters_conv1,
                                                 maxpool=0)
        if self.config.debug == 1:
            print("layer_conv1.shape", layer_conv1.shape)

        layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                                 num_input_channels=self.config.num_filters_conv1,
                                                 conv_filter_size=self.config.filter_size_conv2,
                                                 num_filters=self.config.num_filters_conv2,
                                                 maxpool=0)
        if self.config.debug == 1:
            print("layer_conv2.shape", layer_conv2.shape)


        pool = tf.nn.max_pool(value=layer_conv2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        if self.config.debug == 1:
            print("pool.shape", pool.shape)

        # TODO: this needs to be a deconvolutional layer so to gain back original shape
        # PLEASE sort it out today
        layer_deconv = create_deconvolutional_layer(input=pool,
                                                    num_filters=self.config.num_filters_conv2,
                                                    name='features',
                                                    upscale_factor=2)  # upscale factor 2^# of maxpool
        if self.config.debug == 1:
            print("layer_deconv.shape", layer_deconv.shape)

        self.feature_maps = create_convolutional_layer(input=layer_deconv,
                                                       num_input_channels=self.config.num_filters_conv2,
                                                       conv_filter_size=self.config.initial_im_size,
                                                       num_filters=self.config.num_channels,
                                                       maxpool=0)

        tf.summary.image(name='feature_maps', tensor=self.feature_maps, max_outputs=1)

        if self.config.debug == 1:
            print("features.shape", self.feature_maps.shape)

        # note: this last layer is not meant to learn anything too useful, otherwise the self.feature will not pass on
        # relevant info
        flat_feat = create_flatten_layer(self.feature_maps)

        if self.config.debug == 1:
            print("flat_feat.shape", flat_feat.shape)

        layer_training = create_fc_layer(input=flat_feat,
                                         num_inputs=flat_feat.get_shape()[1:4].num_elements(),
                                         num_outputs=2,
                                         use_relu=True)
        if self.config.debug == 1:
            print("layer_training.shape", layer_training.shape)
        # TODO: make the model trainable
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,
                                                                                               logits=layer_training))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            self.pred = tf.argmax(layer_training, 1)
            correct_prediction = tf.equal(self.pred, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('summaries'):
            self.summaries = tf.summary.merge_all()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
