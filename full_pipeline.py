
from glob import glob
from accessory_functions import sort_nicely
import unet.data_operations as do
from unet.build_unet import unet_network
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
import numpy as np
from skimage.transform import resize, downscale_local_mean
from individual_bacteria_classifier.potential_bacteria_finder import blob_the_builder
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy import ndimage as ndi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Squelch all info messages.


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


def process_gutmask(gutmask):
    """
    Takes the 3D gutmask and processes it to more accurately identify the gut. NOTE: This is written to work with a mask
    that is 1 for the gut and 0 for the background
    :param gutmask: The 3D gutmask
    :return: A processed gutmask
    """
    # Look for regions that are gut in both adjacent frames and include them in the mask
    time_average = gutmask
    for n in range(1, np.shape(gutmask)[0] - 1):
        union = np.floor((gutmask[n-1] + gutmask[n+1]) / 2)
        time_average[n] = np.ceil((union + gutmask[n]) / 2)
    # Fill holes and remove small objects
    time_average = ndi.binary_fill_holes(time_average)
    gutmask = remove_small_objects(time_average, 1000)
    return gutmask


def determine_gutmask(images, load_loc_gutmask, gutmask_region):
    """
    Loads in network hyperparameters, builds network, loads in weights, applies unet to each frame of the scan. Note
    that the gutmask that will be saved is downscaled by a factor of (2, 2).
    :param images: images making up a scan
    :param load_loc_gutmask: the location of the saved model to be used for masking the gut
    :param region: the region of the gut you are in. String, either 'region1' or 'region2'.
    :return: a 3D mask of the gut downscaled (2, 2)
    """
    # Load hyperparameters
    load_loc_gutmask = load_loc_gutmask + '/' + gutmask_region
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
    gutmask = []
    for image in images:
        image = downscale_local_mean(image, (2, 2))  # Hard coded 2 by 2 downsampling
        image = (image - np.mean(image))/np.std(image)
        tiled_image, input_height_original, input_width_original = do.tile_image(image, tile_height, tile_width,
                                                                                 edge_loss)
        predicted_list = []
        for tile in tiled_image:
            prediction = last_layer.eval(feed_dict={input_image_0: [tile]})  # Run through tensorflow graph
            predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction][0]  # convert from softmax to mask
            predicted_list.append(predicted)
        mask = do.detile_image(predicted_list, input_height_original, input_width_original)
        gutmask.append(np.abs(mask - 1))
    session_tf.close()
    gutmask = process_gutmask(gutmask)
    return gutmask


def save_gutmask(save_loc, files_images, gutmask):
    """
    Saves the 3D mask of the gut.
    """
    mask_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'gutmask'
    if not os.path.exists(save_loc + 'gutmasks/'):
        os.mkdir(save_loc + 'gutmasks/')
    np.savez_compressed(save_loc + 'gutmasks/' + mask_name, gutmask=gutmask)


def apply_bacteria_identifier(potential_bacterial_voxels, potential_bacteria_locations, load_loc_bacteria_identifier,
                              bacteria='enterobacter'):
    """
    calls 3D convnet for classifying each (8x28x28) pixel voxel to determine if they are individual bacteria or not.
    :param potential_bacterial_voxels: a list of 30x30x10 images of potential bacteria
    :param potential_bacteria_locations: the x, y, z, locations corresponding to each potential bacteria
    :param load_loc_bacteria_identifier: directory of the saved model for identifying bacteria
    :param bacteria: the type of bacteria you are looking at
    :return: locations of every bacteria and location of every not bacteria
    """
    #                               HYPERPARAMETERS
    tf.reset_default_graph()
    batch_size = 120  # the size of the batches
    initial_kernel = 16  # number of kernels for first layer
    network_depth = 2  # Number of convolutional layers
    final_neurons = 1024  # number of neurons for final dense layer
    kernel_size = [2, 5, 5]  # Size of kernel
    cube_length = 8 * 28 * 28  # flattened size of input image

    #                               CREATE THE TENSORFLOW GRAPH
    flattened_image = tf.placeholder(tf.float32, shape=[None, cube_length])
    input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
    keep_prob = tf.placeholder(tf.float32)
    #   first layer
    output_neurons = cnn_3d(input_image, network_depth=network_depth, kernel_size=kernel_size,
                            num_kernels_init=initial_kernel, keep_prob=keep_prob, final_dense_num=final_neurons)
    prediction = tf.argmax(output_neurons, 1)
    # LOAD PREVIOUS WEIGHTS
    session_tf = tf.InteractiveSession()
    saver_bac = tf.train.Saver()
    saver_bac.restore(session_tf, load_loc_bacteria_identifier + '/' + bacteria + '/model/model.ckpt')
    # noinspection PyPep8,PyPep8
    voxels = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in potential_bacterial_voxels]  #CHANGE?
    predictions = []
    for batch in range(len(voxels) // batch_size):
        offset = batch
        batch_data = voxels[offset:(offset + batch_size)]
        predictions.append(prediction.eval(feed_dict={flattened_image: batch_data, keep_prob: 1.0}))
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


def save_individual_bacteria(save_loc, files_images, bacteria_locs, not_bacteria_locs):
    """
        Save the individual bacteria locations. Saves two different files; location of all of the identified bacteria as
        a list of [[x,y,z], ...] locations and separately all of the potential bacteria that were identified as NOT
        bacteria.
    """
    bacteria_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'bacteria'
    not_bacteria_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'not_bacteria'
    if not os.path.exists(save_loc + 'individual_bacteria/'):
        os.mkdir(save_loc + 'individual_bacteria/')
    if bacteria_locs:  #
        np.savez(save_loc + 'individual_bacteria/' + bacteria_name,
                 bacteria_locs=bacteria_locs)
    if not_bacteria_locs:
        np.savez(save_loc + 'individual_bacteria/' + not_bacteria_name,
                 not_bacteria_locs=not_bacteria_locs)


def determine_aggregate_mask(images, gutmask):
    """
    Creates the 3D mask of all of the bacterial aggregates in the image scan using determine_aggregates.
    :param images: 3D image scan
    :param gutmask: 3D mask of gut
    :return: the 3D mask of the aggregates
    """
    aggregate_mask = np.zeros(np.shape(images))
    for n in range(len(gutmask)):
        temp_mask = remove_small_objects(gutmask[n], 1000)
        labeled_gutmask = label(temp_mask)
        objects = regionprops(labeled_gutmask)
        if len(objects) > 1:
            for object in objects[1:]:
                y_min, x_min, y_max, x_max = [item * 2 for item in object.bbox]
                image = images[n][y_min:y_max, x_min:x_max]
                sub_aggregate_mask = determine_aggregates(image, load_loc_aggregates)
                aggregate_mask[n][y_min:y_max, x_min:x_max] += sub_aggregate_mask
    return aggregate_mask > 0


def determine_aggregates(image, load_loc_aggregates):
    """
    Loads network hyperparameters, builds network, loads saved weights, generates a mask of the aggregates
    :param image: a single image to find aggregates in
    :param load_loc_aggregates: directory of the saved model for finding aggregates
    :return: a single mask of the aggregate
    """
    # Load hyperparameters
    tf.reset_default_graph()
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
    image = (image - np.mean(image))/np.std(image)
    tiled_image, input_height_original, input_width_original = do.tile_image(image, tile_height, tile_width,
                                                                             edge_loss)
    predicted_list = []
    for tile in tiled_image:
        prediction = last_layer.eval(feed_dict={input_image_0: [tile]})  # Run through tensorflow graph
        predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction][0]  # convert from softmax to mask
        predicted_list.append(predicted)
    mask = do.detile_image(predicted_list, input_height_original, input_width_original)
    mask = np.abs(mask - 1)
    session_tf.close()
    return mask


def save_aggregate_mask(save_loc, files_images, aggregate_mask):
    """
    save the 3D mask of all of the detected bacterial aggregates.
    """
    mask_name = files_images[0].split('Scans/')[1].split('pco')[0].replace('/', '_') + 'aggregate_mask'
    if not os.path.exists(save_loc + 'aggregates/'):
        os.mkdir(save_loc + 'aggregates/')
    np.savez_compressed(save_loc + 'aggregates/' + mask_name, gutmask=aggregate_mask)


file_loc = '/media/chiron/Dagobah/deepika/en_ae_invasion'
load_loc_gutmask = '/media/chiron/Stephen Dedalus/automated_pipeline_labels_models/tensorflow_models/gutmask_models/models_for_use'
load_loc_bacteria_identifier = '/media/chiron/Stephen Dedalus/automated_pipeline_labels_models/tensorflow_models/single_bac_models'
load_loc_aggregates = '/media/chiron/Stephen Dedalus/automated_pipeline_labels_models/tensorflow_models/aggregate_model'
bacteria_color_dict = {'488': 'enterobacter', '568': 'aeromonas01'}
region_dict = {'1': 'region_1', '2': 'region_2'}

files_scans = import_files(file_loc)
percent_tracker = 0
for files_images in files_scans:
    # DETERMINE BACTERIAL SPECIES, REGION, AND SAVE LOCATION
    region = files_images[0].split('region_')[-1][0]
    bacterial_species = files_images[0].split('nm/pco')[0][-3:]
    save_loc = files_images[0].split('Scans')[0] + bacteria_color_dict[bacterial_species] + '/'
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)

    # IMPORT IMAGES
    print(str(np.round(percent_tracker * 100 / len(files_scans), 2)) + '% of the data analyzed')
    percent_tracker += 1
    print('importing images')
    images = do.import_images_from_files(files_images, [])

    # FIND AND SAVE GUT MASKS
    print('masking the gut')
    gutmask = determine_gutmask(images, load_loc_gutmask, region_dict[region])
    save_gutmask(save_loc, files_images, gutmask)

    #  FIND AND SAVE INDIVIDUAL BACTERIA
    print('finding individual bacteria')
    potential_bacterial_voxels, potential_bacteria_locations = blob_the_builder(images)
    bacteria_locs, not_bacteria_locs = apply_bacteria_identifier(potential_bacterial_voxels,
                                                                 potential_bacteria_locations,
                                                                 load_loc_bacteria_identifier,
                                                                 bacteria=bacteria_color_dict[bacterial_species])
    save_individual_bacteria(save_loc, files_images, bacteria_locs, not_bacteria_locs)

    # FIND AND SAVE AGGREGATES
    print('finding them bacterial aggregates')
    aggregate_mask = determine_aggregate_mask(images, gutmask)
    save_aggregate_mask(save_loc, files_images, aggregate_mask)
