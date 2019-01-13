import tensorflow as tf
import numpy as np

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters,
                               maxpool=1,
                               name=None):
    with tf.name_scope(name):
        ## We shall define the weights that will be trained using create_weights function.
        weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        ## We create biases using the create_biases function. These are also trained.
        biases = create_biases(num_filters)

        ## Creating the convolutional layer
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        layer += biases

        ## We shall be using max-pooling.
        if maxpool == 1:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        ## Output of pooling is fed to Relu which is the activation function for us.
        layer = tf.nn.relu(layer)

        return layer


def create_deconvolutional_layer(input, num_filters, name, upscale_factor):
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.name_scope(name):
        # Shape of the input tensor
        batch_size = tf.shape(input)[0]
        input_size = input.get_shape().as_list()[1]
        h = input_size * stride
        w = input_size * stride
        # put everything together
        ds = [batch_size]
        ds.append(h)
        ds.append(w)
        ds.append(num_filters)
        deconv_shape = tf.stack(ds)

        filter_shape = [kernel_size, kernel_size, num_filters, num_filters]
        weights = get_bilinear_filter(filter_shape, upscale_factor)

        deconv = tf.nn.conv2d_transpose(value=input,
                                        filter=weights,
                                        output_shape=deconv_shape,
                                        strides=strides,
                                        padding='SAME')
    return deconv


def get_bilinear_filter(filter_shape, upscale_factor):
    # filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ## Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            # Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                        1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def unflatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, 1, 1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def create_convolution(input,
                       num_input_channels,
                       conv_filter_size,
                       num_filters,
                       stride=1,
                       data_format="NHWC"):
    '''
    Simplified version of create_convolutional_layer that doesn't include max pooling or activation function
    :param input: input tensor
    :param num_input_channels: number of input channels
    :param conv_filter_size: width and height of square conv filter
    :param num_filters: number of filters to be applied
    :param stride: scalar stride size for both width and height
    :param data_format: see tensorflow documentation
    :return:
    '''
    if data_format == "NHWC":
        strides = [1, stride, stride, 1]
    elif data_format == "NCHW":
        strides = [1, 1, stride, stride]
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)


    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=strides,
                         padding='SAME')

    layer += biases

    return layer

def summarise_map(name, tensor):
    '''
    Summarises a sliced tensor that has no depth by just adding a depth dim
    :param name: name for the opp
    :param tensor: tensor of shape [batch, h, w]
    :return:
    '''

    expanded_map = tf.expand_dims(tensor, -1)
    tf.summary.image(name=name, tensor=expanded_map, max_outputs=1)


def select_with_matrix_tf(tensor, indexer, batch_size):
    '''

    :param tensor: input tensor of shape [batch, h, w, depth, n_vals]
    :param indexer: indexing tensor of shape [batch, h, w] of ints which
    indicates which layer of the depth dimension to take
    :param batch_size: neccesary since first dim of tensor maybe none and cant set range(None) tensor
    :return: tensor of shape [batch, h, w, n_vals]
    '''
    batch_size_adaptive, h, w, depth, n_vals = tensor.get_shape().as_list()
    desired_shape = [batch_size, h, w]
    index_list = [
        tf.broadcast_to(tf.reshape(tf.range(batch_size, dtype=tf.int64), (-1, 1, 1)), desired_shape),
        tf.broadcast_to(tf.reshape(tf.range(h, dtype=tf.int64), (1, -1, 1)), desired_shape),
        tf.broadcast_to(tf.reshape(tf.range(w, dtype=tf.int64), (1, 1, -1)), desired_shape),
        indexer
    ]
    index = tf.stack(index_list, -1)
    return tf.gather_nd(tensor, index)


def np_sample(labels, label=1, n_samples=180):
    mask = np.zeros(labels.shape)
    if n_samples > 0:
        idx1 = np.where(labels == label)
        idx1 = np.asarray(idx1)
        index = np.random.randint(idx1.shape[1], size=(n_samples))
        a = idx1[:, index]
        mask[a[0], a[1], a[2], a[3]] = label

    return mask
