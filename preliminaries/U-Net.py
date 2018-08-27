
import tensorflow as tf
from time import time
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import re
from scipy import ndimage
from sklearn.model_selection import train_test_split

def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1], strides=[1, 2, 1], padding='SAME')


def convolve(input_image, kernel, num_input_kernels, num_output_kernels):
    weights = tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1)
    bias = tf.constant(0.1, shape=num_output_kernels)
    conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='SAME')
    activation1 = tf.nn.leaky_relu(conv + bias)
    return activation1


def last_convolution(input_image, kernel, num_input_kernels, num_output_kernels):
    weights = tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1)
    bias = tf.constant(0.1, shape=num_output_kernels)
    conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='SAME')
    activation1 = tf.nn.sigmoid(conv + bias)
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


def crop_t1_to_shape_t2(tensor1, tensor2):
    offset0 = (tensor1.shape[1] - tensor2.shape[1]) // 2
    offset1 = (tensor1.shape[2] - tensor2.shape[2]) // 2
    return tensor1[:, offset0:(-offset0), offset1:(-offset1)]


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


def tile_image(input_image):
    # write code to tile the input image and create a list of sub images
    return sub_images


def file_splitter(file):
    return file.split('.')[0].split('/')[-1]



def read_in_images(directory_loc, label_string='_gutmask'):
    files = glob(directory_loc + '/*.tif', recursive=True)
    sort_nicely(files)
    mask_files = [item for item in files if label_string in item]
    data_files = [re.sub('\_gutmask.tif$', '.tif', item) for item in mask_files]  #  insert mask_string ref
    masks = [ndimage.imread(file) for file in mask_files]
    data = [ndimage.imread(file) for file in data_files]
    return train_test_split(data, masks, train_size=0.9)

directory_loc = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/**'
train_data, test_data, train_labels, test_labels = read_in_images(directory_loc)
print(len(test_data))

###  HYPERPARAMETERS
kernel = [3, 3]
num_layers = 5
num_input_images = 1
num_output_images = 16
num_classes = 2
epochs = 120

session_tf = tf.InteractiveSession()

input_image = tf.placeholder(tf.float32, shape=[None, image_size])
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
    cropped = crop_t1_to_shape_t2(down_layers[- (up_iter + 1)], up_conv)
    concatenated = tf.concat(cropped, up_conv)
    num_output_images /= 2
    conv1 = convolve(concatenated, kernel, num_input_images, num_output_images)
    num_input_images = num_output_images
    conv2 = convolve(conv1, kernel, num_input_images, num_output_images)
    up_layers.append(conv2)


last_layer = last_convolution(up_layers[-1], kernel=[1, 1], num_input_kernels=num_output_images,
                              num_output_kernels=num_classes)

prediction = pixel_wise_softmax(last_layer)
loss = dice_loss(prediction, labels)


learning_rate = 0.2
decay_rate = 0.97
momentum = 0.8
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss)
session_tf.run(tf.global_variables_initializer())








train_data = # created flattened list of tiled images (and corresponding flattened list of tiled labels).
train_size = len(train_data)
train_time0 = time()
session_tf.run(tf.global_variables_initializer())
print(str(epochs) + ' epochs')
ac_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    for batch in range(train_size // batch_size):
        offset = (batch * batch_size) % train_size
        batch_data = temp_data[offset:(offset + batch_size)]
        batch_labels = temp_labels[offset:(offset + batch_size)]
        optimizer.run(feed_dict={input_image: batch_data, labels: batch_labels})
        if batch % 500 == 0:
            train_loss = loss.eval(feed_dict={
                input_image: batch_data, labels: batch_labels})
            print("training accuracy %g" % (train_loss))
            ac_list.append(train_loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.plot(ac_list)


