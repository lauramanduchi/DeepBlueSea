from base.base_model import BaseModel
from models.utils_model import *


class MiniVGG(BaseModel):
    def __init__(self, config):
        super(MiniVGG, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, self.config.initial_im_size, self.config.initial_im_size,
                                                   self.config.num_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.initial_im_size, self.config.initial_im_size,
                                                   self.config.num_channels])

        # network architecture
        # two convolutional layers with ReLU activation
        # one maxpool layer
        # two fully convolutional layer
        # one prediction layer
        # one deconvolutional layer
        # one prediction layer

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

        # flatten layer to feed it to fc
        layer_flat = create_flatten_layer(pool)

        if self.config.debug == 1:
            print("layer_flat.shape", layer_flat.shape)

        layer_fc1 = create_fc_layer(input=layer_flat,
                                    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                    num_outputs=self.config.fc_layer_size,
                                    use_relu=True)

        if self.config.debug == 1:
            print("layer_fc1.shape", layer_fc1.shape)

        # dropout during training
        if self.config.train:
            layer_fc1 = tf.nn.dropout(layer_fc1, 0.5)

        layer_fc2 = create_fc_layer(input=layer_fc1,
                                    num_inputs=self.config.fc_layer_size,
                                    num_outputs=self.config.num_features,
                                    use_relu=True)

        if self.config.debug == 1:
            print("layer_fc2.shape", layer_fc2.shape)

        # dropout during training
        if self.config.train:
            layer_fc2 = tf.nn.dropout(layer_fc2, 0.5)

        # prediction layer
        layer_pred = tf.nn.softmax(layer_fc2)

        if self.config.debug == 1:
            print("layer_pred.shape", layer_pred.shape)

        # TODO: make the model trainable
        """
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=layer_deconv1))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            self.pred = tf.argmax(layer_deconv1, 1)
            correct_prediction = tf.equal(self.pred, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        """

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
