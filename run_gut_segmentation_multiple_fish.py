
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
from full_pipeline import process_gutmask, import_files, determine_gutmask, save_gutmask

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Squelch all info messages.


file_loc = '/media/Nilgiri/deepik_invasion_single_time_point/coinoc/AE_EN_di_single_time_point'
load_loc_gutmask = '/media/Stephen/automated_pipeline_labels_models/tensorflow_models/gutmask_models/models_for_use'

bacteria_color_dict = {'488': 'enterobacter', '568': 'aeromonas01'}
region_dict = {'1': 'region_1','2' : 'region_2'}

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
    images,new_labels = do.import_images_from_files(files_images, [], tile=None, edge_loss=0)

    # FIND AND SAVE GUT MASKS
    print('masking the gut')
    gutmask = determine_gutmask(images, load_loc_gutmask, region_dict[region])
    save_gutmask(save_loc, files_images, gutmask)

