

import tensorflow as tf


def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def convolve(input_image, kernel, num_input_kernels, num_output_kernels):
    initializer = tf.contrib.layers.variance_scaling_initializer()
    weights = tf.Variable(initializer([kernel[0], kernel[1], num_input_kernels, num_output_kernels]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='VALID')
    activation1 = tf.nn.leaky_relu(conv + bias)
    return activation1


def up_convolve(input_image, batch_size):  # batch size is hard coded!! It shouldn't be (-1).
    input_shape = input_image.get_shape().as_list()
    num_input_kernels = input_shape[3]
    num_output_kernels = input_shape[3]//2
    output_shape = tf.stack([batch_size, input_shape[1]*2, input_shape[2]*2, num_output_kernels])
    initializer = tf.contrib.layers.variance_scaling_initializer()
    weights = tf.Variable(initializer([2, 2, num_output_kernels, num_input_kernels]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_input_kernels//2]))
    up_conv = tf.nn.conv2d_transpose(input_image, filter=weights, output_shape=output_shape, strides=[1, 2, 2, 1])
    activation1 = tf.nn.leaky_relu(up_conv + bias)
    return activation1


def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


def final_dense_layer(input_image, kernel, num_input_kernels, num_output_kernels):
    initializer = tf.contrib.layers.variance_scaling_initializer()
    weights = tf.Variable(initializer([kernel[0], kernel[1], num_input_kernels, num_output_kernels]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='SAME')
    activation1 = conv + bias
    return activation1


def unet_network(input_tensor, batch_size=2, network_depth=3, kernel_size=[3, 3], num_kernels_init=16,
                 dropout_kept=0.5):
    """
    Builds unet architecture taking in the input tensor and yielding the predicted output tensor. The code is written
    to mimic the initial implementation of unet by Ronneberger et. al.
    :param input_tensor: The tensor of images of shape [batch size, x, y, channels]
    :param batch_size: The size of the batch! Note this is only to inform the up_convolve operation as it didn't like
    the first dimension to be -1
    :param network_depth: Number of layers in depth. There were 4 in the original paper.
    :param kernel_size: Size of convolutional kernels
    :param num_kernels_init: Number of kernels in first convolutional layer, note that this doubles every pooling
    operation as is common.
    :param dropout_kept: The dropout rate is solely implemented in the lowest layer before up-sampling.
    :return: Dictionary containing the predicted output tensor for the batch of input images in one-hot encoding
    [batch, x, y] as well as the states after each convolutional layer (down_states_out, up_states_out)
    """
    down_layers = [input_tensor]
    num_input_images = 1
    num_output_kernels = num_kernels_init
    output_down_states = []
    for down_iter in range(network_depth):
        if all([down_iter > 0, down_iter < network_depth]):
            conv_input_tensor = pool(down_layers[-1])
        else:
            conv_input_tensor = down_layers[-1]
        print('NEW DOWN LAYER')
        conv1 = convolve(conv_input_tensor, kernel_size, num_input_images, num_output_kernels)
        print('conv1: ' + str(conv1.get_shape().as_list()))
        output_down_states.append(conv1)
        num_input_images = num_output_kernels
        conv2 = convolve(conv1, kernel_size, num_input_images, num_output_kernels)
        print('conv2: ' + str(conv2.get_shape().as_list()))
        output_down_states.append(conv2)
        num_output_kernels = num_output_kernels*2
        down_layers.append(conv2)

    #  UP LAYERS
    num_output_kernels = int(num_output_kernels // 2)
    dropout_middle_layer = tf.nn.dropout(down_layers[-1], keep_prob=dropout_kept)
    up_layers = [dropout_middle_layer]
    output_up_states = []
    for up_iter in range(network_depth - 1):
        print('NEW UP LAYER')
        up_conv = up_convolve(up_layers[-1], batch_size=batch_size)
        concatenated = crop_and_concat(down_layers[- (up_iter + 2)], up_conv)
        print('concatenated: ' + str(concatenated.get_shape().as_list()))
        num_output_kernels = int(num_output_kernels // 2)
        conv1 = convolve(concatenated, kernel_size, num_input_images, num_output_kernels)
        print('conv1: ' + str(conv1.get_shape().as_list()))
        output_up_states.append(conv1)
        num_input_images = num_output_kernels
        conv2 = convolve(conv1, kernel_size, num_input_images, num_output_kernels)
        print('conv2: ' + str(conv2.get_shape().as_list()))
        output_up_states.append(conv2)
        up_layers.append(conv2)
    #  FINAL LAYER
    output_tensor = final_dense_layer(up_layers[-1], kernel=[1, 1], num_input_kernels=num_output_kernels,
                                      num_output_kernels=2)
    return {"output": output_tensor, "down_states_out": output_down_states, "up_states_out": output_up_states}
