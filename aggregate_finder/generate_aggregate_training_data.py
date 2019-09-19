
from glob import glob
from accessory_functions import sort_nicely
import unet.data_operations as do
from unet.build_unet import unet_network
import tensorflow as tf
import numpy as np
from skimage.transform import resize, downscale_local_mean
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
import os


def import_files(file_loc):
    """
    Finds all filenames of images in file_loc and groups them by scan.
    :param file_loc: directory containing all images
    :return: A list of lists, each element of files_scans is a list of filenames of images from a single scan
    """
    files = glob(file_loc + '/**/*.tif', recursive=True)
    files.extend(glob(file_loc + '/**/*.png', recursive=True))
    files = [file for file in files if any(['region_1' in file, 'region_2' in file])]
    unique_identifiers = np.unique([file.split('pco')[0] for file in files])
    files_scans = [[file for file in files if unique_identifier in file] for unique_identifier in unique_identifiers]
    for n in range(len(files_scans)):
        sort_nicely(files_scans[n])
    return files_scans


def determine_gutmask(images, load_loc_gutmask):
    """
    Loads in network hyperparameters, builds network, loads in weights, applies unet to each frame of the scan. Note
    that the gutmask that will be saved is downscaled by a factor of (2, 2).
    :param images: images making up a scan
    :param load_loc_gutmask: the location of the saved model to be used for masking the gut
    :return: a 3D mask of the gut downscaled (2, 2)
    """
    # Load hyperparameters
    tf.reset_default_graph()  # This makes sure that the graph is reset to avoid proliferation of open variables.
    hyperparameters = np.load(load_loc_gutmask + '/hyperparameters.npz')
    batch_size = hyperparameters['batch_size']
    initial_kernel = hyperparameters['initial_kernel']
    network_depth = hyperparameters['network_depth']
    tile_height = hyperparameters['tile_height']
    tile_width = hyperparameters['tile_width']

    # Determine the edge loss for the depth of the network
    edge_loss = sum([2 ** (i + 3) for i in range(network_depth)]) - 2 ** (network_depth + 1)
    shape_of_image = [tile_height + edge_loss, tile_width + edge_loss]


    # BUILD UNET
    input_image_0 = tf.placeholder(tf.float32, shape=[None, shape_of_image[0], shape_of_image[1]])
    input_image = tf.reshape(input_image_0, [-1, shape_of_image[0], shape_of_image[1], 1])
    unet_params = unet_network(input_image, batch_size=batch_size, network_depth=network_depth, kernel_size=[3, 3],
                               num_kernels_init=initial_kernel, dropout_kept=1)
    last_layer = unet_params["output"]

    # LOAD PREVIOUS WEIGHTS
    session_tf = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(session_tf, load_loc_gutmask + '/model/model.ckpt')
    print('finished loading model')
    predictions = []
    for image in images:
        image = downscale_local_mean(image, (2, 2))  # Hard coded 2 by 2 downsampling
        image = (image - np.mean(image))/np.std(image)
        tiled_image, input_height_original, input_width_original = do.tile_image(image, tile_height, tile_width,
                                                                                 edge_loss)
        predicted_list = []
        for image in tiled_image:
            prediction = last_layer.eval(feed_dict={input_image_0: [image]})  # Run through tensorflow graph
            predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction][0]  # convert from softmax to mask
            predicted_list.append(predicted)
        mask = do.detile_image(predicted_list, input_height_original, input_width_original)
        predictions.append(np.abs(mask - 1))
    session_tf.close()
    return predictions


file_loc = '/media/rplab/Dagobah/deepika/en_ae_invasion'
load_loc_gutmask = '/media/rplab/Bast/Teddy/gutmask_testing/region1_5_32_downsampled'

files_scans = import_files(file_loc)
files_images = files_scans[2]
for files_images in files_scans:
    images = do.import_images_from_files(files_images)
    # Find gut masks
    gutmask = determine_gutmask(images, load_loc_gutmask)
    gutmask = remove_small_objects(np.array(gutmask, bool), 1000)
    gutmask = remove_small_holes(np.array(gutmask, bool), 10000)
    # apply unet aggregates
    aggregate_mask = np.zeros(np.shape(images))
    for n in range(len(gutmask)):
        temp_mask = remove_small_objects(gutmask[n], 1000)
        labeled_gutmask = label(temp_mask)
        objects = regionprops(labeled_gutmask)
        if len(objects) > 1:
            for object in objects[1:]:
                y_min, x_min, y_max, x_max = [item * 2 for item in object.bbox]
                image = images[n][y_min:y_max, x_min:x_max]

