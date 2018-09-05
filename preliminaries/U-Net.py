


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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolve(input_image, kernel, num_input_kernels, num_output_kernels):
    weights = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='SAME')
    activation1 = tf.nn.leaky_relu(conv + bias)
    return activation1


def last_convolution(input_image, kernel, num_input_kernels, num_output_kernels):
    weights = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], num_input_kernels, num_output_kernels], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_output_kernels]))
    conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding='SAME')
    activation1 = tf.nn.sigmoid(conv + bias)
    return activation1


def up_convolve(input_image):  # The filter and output weights are incorrect on this I think.
    input_shape = input_image.get_shape().as_list()
    num_input_kernels = input_shape[3]
    num_output_kernels = input_shape[3]//2
    output_shape = tf.stack([0, input_shape[1]*2, input_shape[2]*2, num_output_kernels])
    weights = tf.Variable(tf.truncated_normal([2, 2, num_output_kernels, num_input_kernels], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_input_kernels//2]))
    up_conv = tf.nn.conv2d_transpose(input_image, filter=weights, output_shape=output_shape, strides=[1, 2, 2, 1])
    activation1 = tf.nn.leaky_relu(up_conv + bias)
    return activation1


def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


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


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


###  FOR TILING THE DATA  --NEED size2 to be padding so that the output image is of same size of original input image
def tile_image_training(input_image, size2=256, size1=200):  # Tiles an image outputting a list of cut and padded versions of
    # the original image. size1 is the size to which the original image is cut and size2 is the size to which the cut
    # image is padded.
    x1 = [n * size1 for n in range(np.shape(input_image)[0]//size1+1)]
    x2 = x1[1:] + [-1]
    y1 = [n * size1 for n in range(np.shape(input_image)[1]//size1+1)]
    y2 = y1[1:] + [-1]
    sub_images = [np.pad(input_image[x1[splitter]:x2[splitter], y1[splitter2]:y2[splitter2]],
                         (((size2 - np.shape(input_image[x1[splitter]:x2[splitter], y1[splitter2]:y2[splitter2]])[0])//2,
                           (size2 - np.shape(input_image[x1[splitter]:x2[splitter], y1[splitter2]:y2[splitter2]])[0])//2),
                          ((size2 - np.shape(input_image[x1[splitter]:x2[splitter], y1[splitter2]:y2[splitter2]])[1])//2,
                           (size2 - np.shape(input_image[x1[splitter]:x2[splitter], y1[splitter2]:y2[splitter2]])[1])//2)),
                         'reflect')
               for splitter in range(len(x1)) for
              splitter2 in range(len(y1))]
    return sub_images


def read_in_images(directory_loc, label_string='_gutmask', read_in_previous=True, size2=256):
    files = glob(directory_loc + '/*.tif', recursive=True)
    sort_nicely(files)
    mask_files = [item for item in files if label_string in item]
    data_files = [re.sub('\_gutmask.tif$', '.tif', item) for item in mask_files]  #  insert mask_string ref
    if read_in_previous:
        directory_loc = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/previously_labeled'
        files = glob(directory_loc + '/*.tif')
        sort_nicely(files)
        mask_files_2 = [item for item in files if '_mask' in item]
        data_files_2 = [re.sub('\_mask.tif$', '.tif', item) for item in mask_files_2]  #  insert mask_string ref
        mask_files = mask_files + mask_files_2
        data_files = data_files + data_files_2
    masks = [ndimage.imread(file) for file in mask_files]

    tiled_masks = []
    for i in range(len(masks) - 1):
        temp = tile_image_training(masks[i], size2=size2)
        for sub_image in temp:
            sub_image = np.resize(sub_image, (size2, size2))
            tiled_masks.append(sub_image)
    data = [ndimage.imread(file) for file in data_files]
    tiled_data = []
    for i in range(len(data) - 1):
        temp = tile_image_training(data[i])
        for sub_image in temp:
            sub_image = np.resize(sub_image, (size2, size2))
            tiled_data.append(sub_image)
    return train_test_split(tiled_data, tiled_masks, test_size=0.0)



directory_loc = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/**'
subimage_size = [256, 256]
train_data, test_data, train_labels, test_labels = read_in_images(directory_loc, read_in_previous=False)
print(np.shape(train_data))

###  HYPERPARAMETERS
kernel = [3, 3]
num_layers = 3
num_input_images = 1
num_output_images = 16
num_classes = 2
epochs = 120
batch_size = 10
image_size = subimage_size[0] * subimage_size[1]


session_tf = tf.InteractiveSession()
input_image_0 = tf.placeholder(tf.float32, shape=[None, image_size])
input_image = tf.reshape(input_image_0, [-1, subimage_size[0], subimage_size[1], 1])
input_mask_0 = tf.placeholder(tf.float32, shape=[None, image_size])
input_mask = tf.reshape(input_mask_0, [-1, subimage_size[0], subimage_size[1], 1])
down_layers = [input_image]
for down_iter in range(num_layers):
    conv1 = convolve(down_layers[-1], kernel, num_input_images, num_output_images)
    num_input_images = num_output_images
    conv2 = convolve(conv1, kernel, num_input_images, num_output_images)
    pool1 = pool(conv2)
    num_output_images *= 2
    down_layers.append(pool1)


num_output_images = int(num_output_images//2)
up_layers = [down_layers[-1]]
up_iter = 0
for up_iter in range(num_layers):
    up_conv = up_convolve(up_layers[-1])
    concatenated = crop_and_concat(down_layers[- (up_iter + 2)], up_conv)
    num_output_images = int(num_output_images//2)
    conv1 = convolve(concatenated, kernel, num_input_images, num_output_images)
    num_input_images = num_output_images
    conv2 = convolve(conv1, kernel, num_input_images, num_output_images)
    up_layers.append(conv2)




last_layer = last_convolution(up_layers[-1], kernel=[1, 1], num_input_kernels=num_output_images,
                              num_output_kernels=2)  # convert to 1-hot?

prediction = pixel_wise_softmax(last_layer)
loss = dice_loss(prediction, input_mask)
learning_rate = 0.2
decay_rate = 0.97
momentum = 0.8
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss)
session_tf.run(tf.global_variables_initializer())




train_size = len(train_data)
train_time0 = time()
session_tf.run(tf.global_variables_initializer())
print(str(epochs) + ' epochs')
ac_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    temp_data = [image.flatten() for image in train_data]
    temp_labels = [image.flatten() for image in train_labels]
    for batch in range(train_size // batch_size):
        offset = (batch * batch_size) % train_size
        batch_data = temp_data[offset:(offset + batch_size)]
        batch_labels = temp_labels[offset:(offset + batch_size)]
        optimizer.run(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
        if batch % 500 == 0:
            train_loss = loss.eval(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
            print("training accuracy %g" % (train_loss))
            ac_list.append(train_loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.plot(ac_list)


