

import tensorflow as tf
from time import time
import numpy as np
from glob import glob
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import re
from scipy import ndimage
from sklearn.model_selection import train_test_split
from skimage.transform import downscale_local_mean
from skimage import transform
import random
from build_network import unet_network


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


def pixel_wise_softmax(input_tensor):
    exponential_map = tf.exp(input_tensor)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(input_tensor)[3]]))
    return tf.div(exponential_map, tensor_sum_exp)


def dice_loss(prediction, labels):
    print('labels: ' + str(labels.get_shape().as_list()))
    eps = 1e-5
    intersection = tf.reduce_sum(prediction * labels)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
    loss = 1 -(2 * intersection / union)
    return loss


###  DATA PROCESSING  --
def data_augment(images, masks, angle=10, resize_rate=0.9):  # Adjust params. Combine with data read-in in training?
    for i in range(len(masks)):
        image = images[i]/np.amax([np.amax(images[i]), np.abs(np.amin(images[i]))])
        mask = masks[i]
        # should add image resize here

        input_shape_image = image.shape
        input_shape_mask = mask.shape
        size = image.shape[0]
        rsize = random.randint(np.floor(resize_rate * size), size)
        w_s = random.randint(0, size - rsize)
        h_s = random.randint(0, size - rsize)
        sh = random.random() / 2 - 0.25
        rotate_angle = random.random() / 180 * np.pi * angle
        # adjust brightness, contrast, add noise.

        # cropping image
        image = image[w_s:w_s + size, h_s:h_s + size]
        mask = mask[w_s:w_s + size, h_s:h_s + size]
        # affine transform
        afine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angle)  # maybe change this to similarity transform
        image = transform.warp(image, inverse_map=afine_tf, mode='constant', cval=0)
        mask = transform.warp(mask, inverse_map=afine_tf, mode='constant', cval=0)
        # resize to original size
        image = transform.resize(image, input_shape_image, mode='constant', cval=0)
        mask = transform.resize(mask, input_shape_mask, mode='constant', cval=0)
        # should add soft normalize here and simply take in non-normalized images.
        images[i], masks[i] = image, mask
    images = [(sub_image - np.mean(sub_image)) / np.std(sub_image) for sub_image in images]
    return images, masks


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))


def tile_image(input_image, size2=256, size1=200):  # Tiles an image outputting a list of cut and padded versions of
    # the original image. size1 is the size to which the original image is cut and size2 is the size to which the cut
    # image is padded.
    x1 = [n * size1 for n in range(np.shape(input_image)[0]//size1+1)]
    x2 = x1[1:] + [-1]
    y1 = [n * size1 for n in range(np.shape(input_image)[1]//size1+1)]
    y2 = y1[1:] + [-1]
    sub_images = [np.pad(input_image[x1[split_x]:x2[split_x], y1[split_y]:y2[split_y]],
                         (((size2 - np.shape(input_image[x1[split_x]:x2[split_x], y1[split_y]:y2[split_y]])[0])//2,
                           (size2 - np.shape(input_image[x1[split_x]:x2[split_x], y1[split_y]:y2[split_y]])[0])//2),
                          ((size2 - np.shape(input_image[x1[split_x]:x2[split_x], y1[split_y]:y2[split_y]])[1])//2,
                           (size2 - np.shape(input_image[x1[split_x]:x2[split_x], y1[split_y]:y2[split_y]])[1])//2)),
                         'reflect')
               for split_x in range(len(x1)) for
              split_y in range(len(y1))]
    return sub_images


def read_in_images(directory_loc, label_string='_gutmask', read_in_previous=True, size2=256, size1=200, test_size=0.0):
    files = glob(directory_loc + '/*.tif', recursive=True)
    sort_nicely(files)
    mask_files = [item for item in files if label_string in item]
    data_files = [re.sub('\_gutmask.tif$', '.tif', item) for item in mask_files]  #  insert mask_string ref ***
    # mask_files = []
    # data_files = []
    masks = [downscale_local_mean(ndimage.imread(file), (3, 3)) for file in mask_files]
    data = [downscale_local_mean(ndimage.imread(file), (3, 3)) for file in data_files]
    print('data length: ' + str(len(data)))
    if read_in_previous:
        directory_loc = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/previously_labeled_gut_images'
        files = glob(directory_loc + '/*.tif')
        sort_nicely(files)
        mask_files_2 = [item for item in files if '_mask' in item]
        data_files_2 = [re.sub('\_mask.tif$', '.tif', item) for item in mask_files_2]  #  insert mask_string ref  ***
        masks_2 = [ndimage.imread(file) for file in mask_files_2]
        data_2 = [ndimage.imread(file) for file in data_files_2]
        masks = masks + masks_2
        data = data + data_2
    print('data length: ' + str(len(data)))
    # masks = [ndimage.imread(file) for file in mask_files]
    print('done reading in previous masks and data')
    tiled_masks = []
    for i in range(len(masks)):
        temp_masks = tile_image(masks[i], size2=size1, size1=size1)
        for sub_mask in temp_masks:
            sub_mask = np.resize(sub_mask, (size1, size1))
            sub_mask = sub_mask/np.max([np.amax(sub_mask), 1])
            # sub_mask = convert_to_one_hot(sub_image != 0)
            tiled_masks.append(sub_mask)
    # data = [ndimage.imread(file) for file in data_files]
    data = [np.log(image + 1) for image in data]
    data = [(sub_image - np.mean(sub_image)) / np.std(sub_image) for sub_image in data]
    print('done reading in new masks')
    tiled_data = []
    for i in range(len(data)):
        temp_data = tile_image(data[i], size2=size2, size1=size1)
        for sub_image in temp_data:
            sub_image = np.resize(sub_image, (size2, size2))
            tiled_data.append(sub_image)
    print('done reading in new data')
    return train_test_split(tiled_data, tiled_masks, test_size=test_size)


#directory_loc = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/**'
# directory_loc = '/media/parthasarathy/Bast/UNET_Projects/intestinal_outlining/DIC/rough_outline_data/Train/**'
# directory_loc = '/media/parthasarathy/Bast/UNET_Projects/intestinal_outlining/Fluorescence/finished_training_data'
directory_loc = './../data/'

# tiled_image_size = [612, 612]
# cropped_image_size = [572, 572]
tiled_image_size = [412, 412]
cropped_image_size = [372, 372]
train_data, test_data, train_labels, test_labels = read_in_images(directory_loc, label_string='_gutmask', read_in_previous=False,
                                                                  size2=tiled_image_size[0],
                                                                  size1=cropped_image_size[0], test_size=20*10)
print(np.shape(train_data))
print(np.shape(train_labels))


###  HYPERPARAMETERS
num_classes = 2
epochs = 100
batch_size = 8
learning_rate = 0.01
decay_rate = 0.97
decay_steps = 100
momentum = 0.8

session_tf = tf.InteractiveSession()
input_image_0 = tf.placeholder(tf.float32, shape=[None, tiled_image_size[0], tiled_image_size[1]])
input_image = tf.reshape(input_image_0, [-1, tiled_image_size[0], tiled_image_size[1], 1])
input_mask_0 = tf.placeholder(tf.int32, shape=[None, cropped_image_size[0], cropped_image_size[1]])
input_mask = tf.one_hot(input_mask_0, depth=2, on_value=1.0, off_value=0.0, axis=-1)

# BUILD UNET
unet_params = unet_network(input_image, batch_size=batch_size, network_depth=3, kernel_size=[3, 3], num_kernels_init=16,
                          dropout_rate=0.5)
last_layer = unet_params["output"]

###  PREDICTION-LOSS-OPTIMIZER
# prediction = pixel_wise_softmax(last_layer)
# loss = dice_loss(prediction, input_mask)
flat_prediction = tf.reshape(last_layer, [-1, 2])
flat_true = tf.reshape(input_mask, [-1, 2])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_true))
global_step = tf.Variable(0)
learning_rate_decay = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                 decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum=momentum
                                       ).minimize(loss, global_step=global_step)
session_tf.run(tf.global_variables_initializer())

TB = [np.array(test_data[k:k+batch_size]) for k in range(0,batch_size*10,batch_size)]
TL = [np.array(test_labels[k:k+batch_size]) for k in range(0,batch_size*10,batch_size)]

test_batches = [[TB[i],TL[i]] for i in range(len(TB))]

###  TRAINING
train_size = len(train_data)
train_time0 = time()
print(str(epochs) + ' epochs')
ac_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    for batch in range(train_size // batch_size):
        offset = (batch * batch_size) % train_size
        batch_data = np.array(train_data[offset:(offset + batch_size)])
        batch_labels = np.array(train_labels[offset:(offset + batch_size)])
        # batch_data, batch_labels = data_augment(train_data[offset:(offset + batch_size)],
        #                                         train_labels[offset:(offset + batch_size)])
        optimizer.run(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
        if batch % 5 == 0:
            train_loss = loss.eval(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
            test_loss = 0
            for d,l in test_batches:
                test_loss += loss.eval(feed_dict={input_image_0: d, input_mask_0: l})
            test_loss /= len(test_batches)
            print("training loss {}\ttest_loss {}".format(train_loss,test_loss))
            ac_list.append(train_loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.figure()
plt.plot(ac_list)

batch_data = test_batches[0][0]
batch_labels = test_batches[0][1]

# "testing"
prediction = last_layer.eval(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
for incr in range(len(batch_labels)):
    true = batch_labels[incr]
    predicted = [[np.argmax(i) for i in j] for j in prediction[incr]]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(batch_data[incr])
    ax2.imshow(true)
    ax3.imshow(predicted)


plt.savefig("./test_fig.pdf")
