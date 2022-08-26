

from matplotlib import pyplot as plt
from matplotlib import gridspec, patches
import numpy as np
import pickle
from skimage.feature import blob_dog
from skimage.measure import block_reduce
from skimage import exposure
from time import time
from scipy import ndimage
from skimage.feature import peak_local_max
import re
from scipy.ndimage.measurements import center_of_mass, label
import glob
import os.path
from skimage.filters import threshold_otsu
import imageio as io
from skimage import filters

def sort_nicely(l):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def dist(x1, y1, list):
    x2 = list[0]
    y2 = list[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def z_trim_blobs(blobs, directory_loc):
    global image_3D
    image_3D_0 = io.imread(directory_loc[0])

    masks_blobs = [0] * len(directory_loc)

    print('determing 3D blobs')

    for i in range(len(blobs)):
        if len(blobs) > 0:

            sub_z_coordinates = [0] * len(blobs[i])

            for z in range(len(blobs[i])):
                list_coordinates = [[blobs[i][z][0] + k, blobs[i][z][1] + m] for k in range(-5, 5) for m in
                                    range(-5, 5)]
                sub_z_coordinates[z] = list_coordinates

            #### flatten list of lists
            sub_z_coordinates = np.array([item for sublist in sub_z_coordinates for item in sublist])
            temp_masks = np.zeros(np.shape(image_3D_0), dtype=int)

            for b in range(len(sub_z_coordinates)):
                temp_masks[int(sub_z_coordinates[b][0]), int(sub_z_coordinates[b][1])] = 1

            masks_blobs[i] = temp_masks.astype(bool)
        else:
            masks_blobs[i] = np.empty_like(np.shape(image_3D_0), dtype=bool)

    print('labelling bacteria-like objects')
    labels_bacteria = label(masks_blobs)[0]

    image_3D = np.array([io.imread(directory_loc[file]) for file in range(len(directory_loc))])
    region_centers = center_of_mass(np.array(image_3D), labels_bacteria, range(1, np.max(labels_bacteria) + 1))

    #### obtain coordinates as x, y and z.
    coordinates = np.array(
        [[int(region_centers[n][2]), int(region_centers[n][1]), int(region_centers[n][0])] for n in
         range(len(region_centers))])
    image_3D = []
    return coordinates

def upgraded_cube_extractor(ROI_locs, images):
    z_length = 10
    cubeLength = 30
    image_3D = np.array(images)
    cubes = []
    for z_center in ROI_locs:
        if ((z_center[2] > z_length / 2) and (z_center[2] < len(images) - z_length / 2)):
            xstart = int(z_center[0] - cubeLength / 2)
            ystart = int(z_center[1] - cubeLength / 2)
            zstart = int(z_center[2] - z_length / 2)
            subimage = image_3D[zstart: zstart + z_length, xstart:xstart + cubeLength,
                       ystart:ystart + cubeLength].tolist()
            cubes.append(subimage)
    return cubes


def blob_the_builder(images, bacteria_type, direc, region, files_images):
    global blobs
    global start_time
    global plots
    plots = []
    start_time = time()
    blobs = []
    print('starting loop')
    zLength = 10
    sigma = 0.01

    direc_masks = glob.glob(direc.split('Scans')[0] + 'Masks/*.tif')
    print(direc_masks)
    region_mask = [mask for mask in direc_masks if 'region_' + region in mask]
    mask_image = io.imread(region_mask[0])

    gaussian_image = [ndimage.gaussian_filter(file, sigma=sigma) for file in list(images)]
    print('filtered image')

    thresholding_param = threshold_otsu(np.array(gaussian_image))
    print(thresholding_param)
    print('finding local maxima')
    for files in range(len(gaussian_image)):
        coordinates = peak_local_max(gaussian_image[files], min_distance = 30, threshold_abs = thresholding_param,
                                     indices = False, labels = mask_image)  # outputs bool image
        ##### IF THERE ARE SEVERAL MAXIMA IN SAME REGION(multiple pixels with the same maxima adjacent to each other),
        ##### find the center of mass of these regions ####
        labels_image = label(coordinates)[0]
        labelled_centers = center_of_mass(coordinates, labels_image, range(1, np.max(labels_image) + 1))
        temp_blobs = np.array([[int(labelled_centers[n][0]), int(labelled_centers[n][1])] for n in range(len(labelled_centers))])
        blobs.append(temp_blobs)

        print('Finding blobs in image ' + str(files) + ' of ' + str(len(gaussian_image)))

    new_blobs = [0] * len(gaussian_image)
    for files in range(len(blobs)):
        list_blobs_Z = [[blobs[files][b][0], blobs[files][b][1], 1, 0, 0] for b in range(len(blobs[files]))]
        new_blobs[files] = list_blobs_Z

    blobs = new_blobs

    final_centers = z_trim_blobs(blobs, files_images)

    ROI_locs = [list(i) for i in final_centers]
    ROI_locs = [[ROI_locs[i][1], ROI_locs[i][0], ROI_locs[i][2]] for i in range(len(ROI_locs)) if
                (ROI_locs[i][2] > zLength / 2 and ROI_locs[i][2] < len(files_images) - zLength / 2)
                ]
    blob_locs = list(sorted(ROI_locs, key=lambda x: x[2]))

    # Extract a cube around each blob for classification
    potential_bacteria_voxels = upgraded_cube_extractor(blob_locs, images)
    print('finished extracting cubes')

    potential_bacteria_voxels = [(image - np.mean(image)) / np.std(image) for image in potential_bacteria_voxels]
    return potential_bacteria_voxels, blob_locs
