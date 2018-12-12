

import tensorflow as tf
import unet.data_processing as dp
import unet.training as train
from time import time
from matplotlib import pyplot as plt
from unet.build_network import unet_network
import numpy as np
from glob import glob
from scipy import ndimage
from skimage.transform import downscale_local_mean


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



###  HYPERPARAMETERS
num_classes = 2
epochs = 2
batch_size = 2
learning_rate = 0.01
decay_rate = 1
decay_steps = 100
momentum = 0.9
network_depth = 3
tiled_image_size = [412, 412]
edge_loss_dict = {'3': 40, '4': 88}
cropped_image_size = [i - edge_loss_dict[str(network_depth)] for i in tiled_image_size]


# Determine local drive, decide whether to load or save the weights
drive = dp.drive_loc('Bast')
save, save_loc = False, drive + '/Teddy/tf_models/DIC_rough_outline/model.ckpt'
load, load_loc = False, drive + '/Teddy/tf_models/DIC_rough_outline/model.ckpt'
# directory_loc = drive + '/UNET_Projects/intestinal_outlining/DIC/kristis_training_images/tmp'
directory_loc = drive + '/UNET_Projects/intestinal_outlining/DIC/rough_outline_data/Train'
train_data, test_data, train_labels, test_labels = dp.read_in_images(directory_loc, label_string='_mask',
                                                                  size2=tiled_image_size[0],
                                                                  size1=cropped_image_size[0], test_size=0.1,
                                                                  import_length=4000)


# BUILD UNET
session_tf = tf.InteractiveSession()
input_image_0 = tf.placeholder(tf.float32, shape=[None, tiled_image_size[0], tiled_image_size[1]])
input_image = tf.reshape(input_image_0, [-1, tiled_image_size[0], tiled_image_size[1], 1])
input_mask_0 = tf.placeholder(tf.int32, shape=[None, cropped_image_size[0], cropped_image_size[1]])
input_mask = tf.one_hot(input_mask_0, depth=2, on_value=1.0, off_value=0.0, axis=-1)
unet_params = unet_network(input_image, batch_size=batch_size, network_depth=network_depth, kernel_size=[3, 3],
                           num_kernels_init=16, dropout_kept=0.8)
last_layer = unet_params["output"]

###  PREDICTION-LOSS-OPTIMIZER
optimizer, loss = train.optimizer_func(last_layer, input_mask, momentum=momentum, learning_rate=learning_rate,
                                       decay_steps=decay_steps, decay_rate=decay_rate)
session_tf.run(tf.global_variables_initializer())

# LOAD PREVIOUS WEIGHTS
if load:
    session_tf = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(session_tf, load_loc)
    print('finished loading model')

###  TRAINING
train_size = len(train_data)
train_time0 = time()
print(str(epochs) + ' epochs')
ac_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    for batch in range(train_size // batch_size):
        offset = (batch * batch_size) % train_size
        batch_data = train_data[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]
        batch_data, batch_labels = dp.data_augment(train_data[offset:(offset + batch_size)],
                                                   train_labels[offset:(offset + batch_size)])
        optimizer.run(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
        if batch % 5 == 0:
            train_loss = loss.eval(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
            print("training loss %g" % (train_loss))
            ac_list.append(train_loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.figure()
plt.plot(ac_list)

# SAVE UNET
if save:
    saver = tf.train.Saver()
    save_path = saver.save(session_tf, save_loc)
    print("Model saved in path: %s" % save_path)



from skimage import io
file_loc = drive + '/UNET_Projects/intestinal_outlining/DIC/test_full_scan/test/'
save_loc = drive + '/UNET_Projects/intestinal_outlining/DIC/test_full_scan/prediction_2.0/'
files = glob(file_loc + '*.tif')
files = [file for file in files if 'mask' not in file]
dp.sort_nicely(files)
file = files[0]
for file in files:
    image = downscale_local_mean(ndimage.imread(file), (2, 2))
    image = (image - np.mean(image)) / np.std(image)
    size1, size2 = 372, 412
    images = dp.tile_image(image, size1=size1, size2=size2)
    images = [np.resize(image, (412, 412)) for image in images]
    # "testing"
    prediction = last_layer.eval(feed_dict={input_image_0: images})
    predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction]
    predicted = dp.detile_1(image, predicted)
    io.imsave(save_loc + file.split('/')[-1],
              np.array(np.concatenate((image, abs(predicted)*image), axis=1), dtype='float32'))
    # f, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(image)
    # ax2.imshow(predicted)
print('done saving')

plt.imshow(np.concatenate((image, abs(predicted-1)*image), axis=1))

