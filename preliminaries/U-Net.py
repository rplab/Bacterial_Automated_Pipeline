
import tensorflow as tf


def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1], strides=[1, 2, 1], padding='SAME')


def convolve(input_image, kernel, num_input_kernels, num_output_kernels):
    weights = tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1)
    bias = tf.constant(0.1, shape=num_output_kernels)
    conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='SAME')
    activation1 = tf.nn.leaky_relu(conv + bias)
    return activation1


def up_convolve(input_image):
    input_shape = tf.shape(input_image)
    num_output_kernels = input_shape[3]//2
    num_input_kernels = input_shape[0]
    output_shape = tf.stack([num_input_kernels, input_shape[1]*2, input_shape[2]*2, num_output_kernels])
    weights = tf.truncated_normal([kernel[0], kernel[1], num_output_kernels, num_input_kernels], stddev=0.1)
    bias = tf.constant(0.1, shape=num_input_kernels//2)
    up_conv = tf.nn.conv2d_transpose(input_image, filter=weights, output_shape=output_shape, strides=[1, 1, 1, 1])
    activation1 = tf.nn.leaky_relu(up_conv + bias)
    return activation1


def dense_layer(input_image, num_input_kernels, num_output_neurons):
    weights = tf.truncated_normal([num_input_kernels, num_output_neurons], stddev=0.1)
    bias = tf.constant(0.1, shape=num_output_neurons)
    flattened_image = tf.reshape(input_image, [-1, num_input_kernels])  # flatten the input image
    dense =  tf.nn.leaky_relu(tf.matmul(flattened_image, weights) + bias)
    return dense


def pixel_wise_softmax(last_layer):
    exponential_map = tf.exp(last_layer)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(last_layer)[3]]))
    return tf.div(exponential_map, tensor_sum_exp)


def dice_loss(prediction, labels):
    smooth_factor = 1
    intersection = tf.reduce_sum(prediction * labels) + smooth_factor
    union = tf.reduce_sum(prediction) + tf.reduce_sum(labels) + smooth_factor
    loss = -(2 * intersection / union)
    return loss


###  HYPERPARAMETERS
kernel = [3, 3]
num_layers = 5
num_input_images = 1
num_output_images = 16
num_classes = 2
input_image
labels


down_layers = [input_image]
for down_iter in range(num_layers):
    conv1 = convolve(down_layers[-1], kernel, num_input_images, num_output_images)
    num_input_images = num_output_images
    conv2 = convolve(conv1, kernel, num_input_images, num_output_images)
    pool1 = pool(conv2)
    num_output_images *= 2
    down_layers.append(pool1)

up_layers = [down_layers[-1]]
for up_iter in range(num_layers):
    up_conv = up_convolve(up_layers[-1])
    concatenated = tf.concat(down_layers[- (up_iter + 1)], up_conv)
    num_output_images /= 2
    conv1 = convolve(concatenated, kernel, num_input_images, num_output_images)
    num_input_images = num_output_images
    conv2 = convolve(conv1, kernel, num_input_images, num_output_images)
    up_layers.append(conv2)

last_layer = convolve(up_layers[-1], kernel=[1, 1], num_input_kernels=num_output_images, num_output_kernels=num_classes)

prediction = pixel_wise_softmax(last_layer)
loss = dice_loss(prediction, labels)

