
from glob import glob
from accessory_functions import sort_nicely
import unet.data_operations as do
from unet.build_unet import unet_network
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from individual_bacteria_classifier.potential_bacteria_finder import blob_the_builder
from skimage.measure import label, regionprops
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
    Loads in network hyperparameters, builds network, loads in weights, applies unet to each frame of the scan
    :param images: images making up a scan
    :param load_loc_gutmask: the location of the saved model to be used for masking the gut
    :return: a 3D mask of the gut in one hot notation
    """
    # Load hyperparameters
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
        tiled_image, input_height_original, input_width_original = do.tile_image(image, tile_height, tile_width,
                                                                                 edge_loss)
        predicted_list = []
        for image in tiled_image:
            prediction = last_layer.eval(feed_dict={input_image_0: [image]})  # Run through tensorflow graph
            predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction][0]  # convert from softmax to mask
            predicted_list.append(predicted)
        mask = do.detile_image(predicted_list, input_height_original, input_width_original)
        predictions.append(mask)
    session_tf.close()
    return predictions


def determine_aggregates(image, load_loc_aggregates):
    """
    Loads network hyperparameters, builds network, loads saved weights, generates a mask of the aggregates
    :param image: a single image to find aggregates in
    :param load_loc_aggregates: directory of the saved model for finding aggregates
    :return: a single mask of the aggregate
    """
    # Load hyperparameters
    hyperparameters = np.load(load_loc_aggregates + '/hyperparameters.npz')
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
    saver.restore(session_tf, load_loc_aggregates + '/model/model.ckpt')
    print('finished loading model')
    tiled_image, input_height_original, input_width_original = do.tile_image(image, tile_height, tile_width,
                                                                             edge_loss)
    predicted_list = []
    for tile in tiled_image:
        prediction = last_layer.eval(feed_dict={input_image_0: [tile]})  # Run through tensorflow graph
        predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction][0]  # convert from softmax to mask
        predicted_list.append(predicted)
    mask = do.detile_image(predicted_list, input_height_original, input_width_original)
    session_tf.close()
    return mask


def apply_bacteria_identifier(potential_bacterial_voxels, potential_bacteria_locations, load_loc_bacteria_identifier,
                              bacteria='enterobacter'):
    """

    :param potential_bacterial_voxels: a list of 30x30x10 images of potential bacteria
    :param potential_bacteria_locations: the x, y, z, locations corresponding to each potential bacteria
    :param load_loc_bacteria_identifier: directory of the saved model for identifying bacteria
    :param bacteria: the type of bacteria you are looking at
    :return: locations of every bacteria and location of every not bacteria
    """
    #                               HYPERPARAMETERS

    depth = 2  # Number of convolutional layers
    L1 = 16  # number of kernels for first layer
    L_final = 1024  # number of neurons for final dense layer
    kernel_size = [2, 5, 5]  # Size of kernel
    batch_size = 120  # the size of the batches
    cube_length = 8 * 28 * 28  # flattened size of input image

    #                               CREATE THE TENSORFLOW GRAPH
    flattened_image = tf.placeholder(tf.float32, shape=[None, cube_length])
    input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
    keep_prob = tf.placeholder(tf.float32)
    #   first layer
    output_neurons = cnn_3d(input_image, network_depth=depth, kernel_size=kernel_size, num_kernels_init=L1,
                            keep_prob=keep_prob,
                            final_dense_num=L_final)
    output_neurons.get_shape()
    prediction = tf.argmax(output_neurons, 1)
    # LOAD PREVIOUS WEIGHTS

    bacteria = 'enterobacter'
    load_loc_bacteria_identifier = '/media/rplab/Bast/Teddy/single_bac_labeled_data/single_bac_models'

    session_tf = tf.InteractiveSession()
    saver_bac = tf.train.Saver()
    saver_bac.restore(session_tf, load_loc_bacteria_identifier + '/' + bacteria + '/model/model.ckpt')
    voxels = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in potential_bacterial_voxels]
    predictions = []
    for batch in range(len(voxels) // batch_size):
        offset = batch
        print(offset)
        batch_data = voxels[offset:(offset + batch_size)]
        predictions.append(prediction.eval(feed_dict={flattened_image: batch_data, keep_prob: 1.0})[0])
    predictions = np.array(predictions).flatten()
    session_tf.close()
    bacterial_locs = []
    not_bacterial_locs = []
    for n in range(len(predictions)):
        if predictions[n] == 1:
            bacterial_locs.append(potential_bacteria_locations[n])
        else:
            not_bacterial_locs.append(potential_bacteria_locations)
    return bacterial_locs, not_bacterial_locs


file_loc = '/media/rplab/Dagobah/deepika/en_ae_invasion'
load_loc_gutmask = '/media/rplab/Bast/Teddy/gutmask_testing/region1_5_32_downsampled'
load_loc_bacteria_identifier = '/media/rplab/Bast/Teddy/single_bac_labeled_data/single_bac_models'
load_loc_aggregates = '/media/rplab/Bast/Teddy/aggregate_testing/bac_aggregate_model'
bacteria_color_dict = {'488': 'enterobacter', '568': 'aeromonas'}

files_scans = import_files(file_loc)
print(files_scans[0])
files_images = files_scans[0]
for files_images in files_scans:
    images = do.import_images_from_files(files_images)
    print(np.shape(images))
    # Find gut masks
    gutmask = determine_gutmask(images, load_loc_gutmask)
    mask_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'gutmask'
    os.mkdir(files_images[0].split('Scans')[0] + 'gutmasks/')
    np.savez_compressed(files_images[0].split('Scans')[0] + 'gutmasks/' + mask_name, gutmask=gutmask)
    # test this on two scans, possible tensorflow graph issues and tensorflow session issues.

    #  Find individual bacteria
    potential_bacterial_voxels, potential_bacteria_locations = blob_the_builder(images)
    bacterial_species = files_images[0].split('nm/pco')[0][-3:]
    bacteria_locs, not_bacteria_locs = apply_bacteria_identifier(potential_bacterial_voxels,
                                                                 potential_bacteria_locations,
                                                                 load_loc_bacteria_identifier,
                                                                 bacteria=bacteria_color_dict[bacterial_species])
    bacteria_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'bacteria'
    not_bacteria_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'not_bacteria'
    os.mkdir(files_images[0].split('Scans')[0] + 'individual_bacteria/')
    np.savetxt(files_images[0].split('Scans')[0] + 'individual_bacteria/' + bacteria_name, np.array(bacteria_locs))
    np.savetxt(files_images[0].split('Scans')[0] + 'individual_bacteria/' + not_bacteria_name, np.array(not_bacteria_locs))

    # apply unet aggregates
    aggregate_mask = np.zeros(np.shape(images))
    for n in range(len(gutmask)):
        labeled_gutmask = label(gutmask[n])
        objects = regionprops(labeled_gutmask)
        for object in objects:
            y_min, x_min, y_max, x_max = object.bbox()
            image = images[n][y_min:y_max, x_min:x_max]
            sub_aggregate_mask = determine_aggregates(image, load_loc_aggregates)
            aggregate_mask[n][y_min:y_max, x_min:x_max] += sub_aggregate_mask
    aggregate_mask = aggregate_mask > 0
    mask_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'aggregate_mask'
    os.mkdir(files_images[0].split('Scans')[0] + 'aggregates/')
    np.savez_compressed(files_images[0].split('Scans')[0] + 'aggregates/' + mask_name, gutmask=aggregate_mask)


