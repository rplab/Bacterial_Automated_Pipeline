
import tensorflow as tf
from matplotlib import pyplot as plt
from unet.build_unet import unet_network
import numpy as np
from unet import data_operations as do


save_figures = False
use_masks = False


# Set locations to load images, weights, and hyperparameters
load_loc = '/home/chiron/Documents/DIC_gut_segmentation_saves/with_julia_data'
file_loc = '/home/chiron/Documents/DIC_lumen_not_lumen_data_downsampled/test'


# Load hyperparameters
hyperparameters = np.load(load_loc + '/hyperparameters.npz')
num_classes = hyperparameters['num_classes']
batch_size = hyperparameters['batch_size']
initial_kernel = hyperparameters['initial_kernel']
network_depth = hyperparameters['network_depth']
tile_height = hyperparameters['tile_height']
tile_width = hyperparameters['tile_width']
downscale = hyperparameters['downscale']


# Determine the edge loss for the depth of the network
edge_loss = sum([2**(i+3) for i in range(network_depth)]) - 2**(network_depth+1)
cropped_image_size = [tile_height, tile_width]
shape_of_image = [tile_height + edge_loss, tile_width + edge_loss]


# Load images
if use_masks:
    test_data_files, test_label_files = do.get_training_files(file_loc)
else:
    test_data_files = do.get_files(file_loc)
    test_label_files = []
test_data, test_labels = do.import_images_from_files(test_data_files, test_label_files, downscale=downscale,
                                                     tile=None, edge_loss=0)


# BUILD UNET
input_image_0 = tf.placeholder(tf.float32, shape=[None, shape_of_image[0], shape_of_image[1]])
input_image = tf.reshape(input_image_0, [-1, shape_of_image[0], shape_of_image[1], 1])
unet_params = unet_network(input_image, batch_size=batch_size, network_depth=network_depth, kernel_size=[3, 3],
                           num_kernels_init=initial_kernel, dropout_kept=1)
last_layer = unet_params["output"]


# LOAD PREVIOUS WEIGHTS
session_tf = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(session_tf, load_loc + '/model/model.ckpt')
print('finished loading model')


# Test images
predictions = []
for n in range(len(test_data)):
    tiled_image, input_height_original, input_width_original = do.tile_image(test_data[n], tile_height, tile_width,
                                                                             edge_loss)
    predicted_list = []
    for image in tiled_image:
        prediction = last_layer.eval(feed_dict={input_image_0: [image]})  # Run through tensorflow graph
        predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction][0]  # convert from softmax to mask
        predicted_list.append(predicted)
    mask = do.detile_image(predicted_list, input_height_original, input_width_original)
    predictions.append(mask)


# Display the first image
alpha = 0.4
n = 0
cmap = 'gray'
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.imshow(test_data[n], cmap=cmap)
ax1.axis('off')
ax2.imshow(test_data[n], cmap=cmap)
if test_labels:
    ax2.imshow(test_labels[n], alpha=alpha, cmap=cmap)
ax2.axis('off')
ax3.imshow(test_data[n], cmap=cmap)
ax3.imshow(predictions[n], alpha=alpha, cmap=cmap)
ax3.axis('off')

if save_figures:
    for m in range(len(predictions)):
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(test_data[m], cmap=cmap)
        ax1.axis('off')
        ax2.imshow(test_data[m], cmap=cmap)
        ax2.imshow(predictions[m], alpha=alpha, cmap=cmap)
        ax3.axis('off')

        plt.savefig(load_loc + '/predictions/figure' + str(m), format='png')
        plt.close()
