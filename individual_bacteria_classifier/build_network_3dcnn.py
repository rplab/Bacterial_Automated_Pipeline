

import tensorflow as tf
from functools import reduce



def pool(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolve(input_image, kernel, num_input_kernels, num_output_kernels, is_train):
    weights = tf.Variable(tf.random.truncated_normal([kernel[0], kernel[1], kernel[2], num_input_kernels, num_output_kernels],
                                              stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    conv = tf.nn.conv3d(input_image, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
    # conv_normed = tf.layers.batch_normalization(conv + bias, training=is_train)
    activation1 = tf.nn.leaky_relu(conv + bias)
    return activation1


def weightVariable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def denseLayer(x, numIn=2 * 7 * 7 * 32, numOut=1024):
    W_fc1 = weightVariable([numIn, numOut])
    b_fc1 = biasVariable([numOut])
    h_pool2_flat = tf.reshape(x, [-1, numIn])
    dense = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return dense


def softmaxLayer(x, numIn=1024, numLabels=2):
    W_fc2 = weightVariable([numIn, numLabels])
    b_fc2 = biasVariable([numLabels])
    return tf.nn.softmax(tf.matmul(x, W_fc2) + b_fc2)


def cnn_3d(input_tensor, network_depth=3, kernel_size=[2, 5, 5], num_kernels_init=16, keep_prob=0.5,
           final_dense_num=1024, is_train=True):
    down_layers = [input_tensor]
    num_input_images = 1
    num_output_images = num_kernels_init
    for down_iter in range(network_depth):
        conv_input_tensor = down_layers[-1]
        conv1 = convolve(conv_input_tensor, kernel_size, num_input_images, num_output_images, is_train)
        num_input_images = num_output_images
        ###   UNCOMMENT THE FOLLOWING TO INCLUDE AN EXTRA CONVOLUTION PRE POOLING ###
        # conv2 = convolve(conv1, kernel_size, num_input_images, num_output_images)
        # print('conv2: ' + str(conv2.get_shape().as_list()))
        # output_down_states.append(conv2)
        num_output_images *= 2
        pooled = pool(conv1)
        down_layers.append(pooled)
    num_neurons = reduce(lambda x, y: x*y, (down_layers[-1]).get_shape().as_list()[1:])
    final_dense = denseLayer(down_layers[-1], numIn=num_neurons, numOut=final_dense_num)
    dropped = tf.nn.dropout(final_dense, 1 - (keep_prob))
    soft_max = softmaxLayer(dropped, numIn=final_dense_num, numLabels=2)
    return soft_max

