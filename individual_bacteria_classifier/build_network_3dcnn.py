

import tensorflow as tf


def pool(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolve(input_image, kernel, num_input_kernels, num_output_kernels):
    weights = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    conv = tf.nn.conv3d(input_image, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
    conv_normed = tf.layers.batch_normalization(conv + bias)
    activation1 = tf.nn.leaky_relu(conv_normed)
    return activation1

def final_dense_layer(input_image, kernel, num_input_kernels, num_output_kernels):
    weights = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    conv = tf.nn.conv3d(input_image, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
    activation1 = tf.nn.leaky_relu(conv + bias)
    return activation1


def softmaxLayer(x, kernel, num_input_kernels, num_output_kernels):
    weights = tf.Variable(
        tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    return tf.nn.softmax(tf.matmul(x, weights) + bias)


def cnn_3d(input_tensor, network_depth=3, kernel_size=[3, 3], num_kernels_init=16, keep_prob=0.5,
           final_dense_num=1024):
    down_layers = [input_tensor]
    num_input_images = 1
    num_output_images = num_kernels_init
    output_down_states = []
    for down_iter in range(network_depth):
        if all([down_iter > 0, down_iter < network_depth]):
            conv_input_tensor = pool(down_layers[-1])
        else:
            conv_input_tensor = down_layers[-1]
        print('NEW DOWN LAYER')
        conv1 = convolve(conv_input_tensor, kernel_size, num_input_images, num_output_images)
        print('conv1: ' + str(conv1.get_shape().as_list()))
        output_down_states.append(conv1)
        num_input_images = num_output_images
        conv2 = convolve(conv1, kernel_size, num_input_images, num_output_images)
        print('conv2: ' + str(conv2.get_shape().as_list()))
        output_down_states.append(conv2)
        num_output_images *= 2
        down_layers.append(conv2)
    final_dense = final_dense_layer(down_layers[-1], kernel=[1, 1], num_input_kernels=num_output_images,
                                    num_output_kernels=final_dense_num)
    dropped = tf.nn.dropout(final_dense, keep_prob)
    soft_max = softmaxLayer(dropped, kernel=[1, 1], num_input_kernels=num_output_images,
                            num_output_kernels=2)
    return {"output": soft_max, "down_states_out": output_down_states}

