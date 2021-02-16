
from glob import glob
from accessory_functions import sort_nicely
import unet.data_operations as do
from unet.build_unet import unet_network
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
import numpy as np
from skimage.transform import resize, downscale_local_mean, rescale
from individual_bacteria_classifier.potential_bacteria_finder import blob_the_builder
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, binary_erosion
from scipy import ndimage as ndi
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Squelch all info messages.

def import_files(file_loc):
    """
    Finds all filenames of images in file_loc and groups them by scan.
    :param file_loc: directory containing all images
    :return: A list of lists, each element of files_scans is a list of filenames of images from a single scan
    """
    files = glob(file_loc + '/**/*.tif', recursive=True)
    files.extend(glob(file_loc + '/**/*.png', recursive=True))
    files = [file for file in files if any(['region_1' in file, 'region_2' in file]) and 'Masks' not in file]
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

    normalize_val = 1 / (np.max(gutmask) - np.min(gutmask))
    min_value = np.min(gutmask)
    normalized_gut_mask = (gutmask - min_value) * normalize_val
    bool_gutmask = 1 - np.floor(normalized_gut_mask)

    average_mask = bool_gutmask

    for n in range(1, len(average_mask)-1):
        union = np.floor((average_mask[n-1] + average_mask[n+1]) / 2)
        bool_gutmask[n] = np.ceil((union + average_mask[n]) / 2)

    erode_mask_fill_holes = [ndi.binary_fill_holes(binary_erosion(bool_gutmask[i])) for i in range(len(bool_gutmask))]
    gutmask = [remove_small_objects(erode_mask_fill_holes[i], 5000) for i in range(len(erode_mask_fill_holes))]

    # Fill holes and remove small objects
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
    tf.compat.v1.reset_default_graph()  # This makes sure that the graph is reset to avoid proliferation of open variables.
    hyperparameters = np.load(load_loc_gutmask + '/hyperparameters.npz')
    batch_size = hyperparameters['batch_size']
    initial_kernel = hyperparameters['initial_kernel']
    network_depth = hyperparameters['network_depth']
    tile_height = hyperparameters['tile_height']
    tile_width = hyperparameters['tile_width']
    downscale = hyperparameters['downscale']

    if np.shape(downscale):
        downscale = tuple(downscale)
    # Determine the edge loss for the depth of the network
    edge_loss = sum([2 ** (i + 3) for i in range(network_depth)]) - 2 ** (network_depth + 1)
    shape_of_image = [tile_height + edge_loss, tile_width + edge_loss]


    # BUILD UNET
    input_image_0 = tf.compat.v1.placeholder(tf.float32, shape=[None, shape_of_image[0], shape_of_image[1]])
    input_image = tf.reshape(input_image_0, [-1, shape_of_image[0], shape_of_image[1], 1])
    unet_params = unet_network(input_image, batch_size=batch_size, network_depth=network_depth, kernel_size=[3, 3],
                               num_kernels_init=initial_kernel, dropout_kept=1)
    last_layer = unet_params["output"]

    # LOAD PREVIOUS WEIGHTS
    session_tf = tf.compat.v1.InteractiveSession()
    saver = tf.compat.v1.train.Saver()
    saver.restore(session_tf, load_loc_gutmask + '/model/model.ckpt')
    gutmask = []
    for image in images:
        image = downscale_local_mean(image,  downscale[0:2])
        image = (image - np.mean(image))/np.std(image)
        tiled_image, input_height_original, input_width_original = do.tile_image(image, tile_height, tile_width,
                                                                                 edge_loss)
        predicted_list = []
        for tile in tiled_image:
            prediction = last_layer.eval(feed_dict={input_image_0: [tile]})  # Run through tensorflow graph
            predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction][0]  # convert from softmax to mask
            predicted_list.append(predicted)
        mask = do.detile_image(predicted_list, input_height_original, input_width_original)
        mask = rescale(mask, 2, anti_aliasing=False)
        gutmask.append(mask)
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


filez_loc = '/media/rplab/Aravalli/AEMB_EN_invasion_time_series/12_1_2020_ts_AEMB_dtom_EN_gfp/'
load_loc_gutmask = '/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/tensorflow_models/gutmask_models/models_for_use'

bacteria_color_dict = {'488': 'enterobacter', '568': 'aeromonas_mb'}
region_dict = {'1': 'region_1','2' : 'region_2'}

files_scans = import_files(filez_loc)
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
    images,new_labels = do.import_images_from_files(files_images, [], tile=None, edge_loss=0)

    # FIND AND SAVE GUT MASKS
    print('masking the gut')
    gutmask = determine_gutmask(images, load_loc_gutmask, region_dict[region])
    save_gutmask(save_loc, files_images, gutmask)

