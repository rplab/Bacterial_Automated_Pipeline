

from skimage import transform
from skimage.transform import downscale_local_mean
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import random
from glob import glob
import re
import imageio
from scipy import ndimage as ndi
from accessory_functions import sort_nicely
import imageio as io

def shuffle(images, masks):
    """
    Zips images and masks together, randomly shuffles the order, then unzips to two lists again.
    :param images: images to be shuffled
    :param masks: masks to be shuffled
    :return: images and masks with order randomly shuffled, but still matching between the two
    """
    zipped = list(zip(images, masks))
    random.shuffle(zipped)
    images, masks = zip(*zipped)
    return images, masks


def get_files(file_loc):
    """
    Retrieves file names for all images and masks.
    :param file_loc: Location of parent directory for images/masks
    :return: Outputs lists of file names for images
    """
    # Determine desired files to import
    files = glob(file_loc + '/*.tif', recursive=True)
    files.extend(glob(file_loc + '/*.png', recursive=True))
    sort_nicely(files)
    return files


def get_training_files(file_loc):
    """
    Retrieves file names for all images and masks.
    :param file_loc: Location of parent directory for images/masks
    :return: Outputs lists of file names for images and masks
    """
    # Determine desired files to import
    label_string = '_mask'
    files = glob(file_loc + '/*.tif', recursive=True)
    files.extend(glob(file_loc + '/*.png', recursive=True))
    sort_nicely(files)
    mask_files = [item for item in files if label_string in item]
    image_files = [re.sub(label_string, '', item) for item in mask_files]

    return image_files, mask_files


def pad_images(images, pad_to):
    if pad_to != 0:
        images_out = [np.pad(image, ((pad_to, pad_to), (pad_to, pad_to)), mode='reflect') for image in images]
    else:
        images_out = images
    return images_out


def data_augment(images, masks, max_angle=10, resize_ratio=0.95):
    """
    Runs various data augmentation methods on a set of input images and their masks
    :param images: Images to be modified
    :param masks: Masks to go with the images
    :param max_angle: The maximum angle to randomly shift images and masks by
    :param resize_ratio: minimum percent to scale the image down in both dimensions
    :return: Altered images and masks
    """
    images_out = []
    masks_out = []
    for i in range(len(masks)):
        image = images[i]/np.amax([np.amax(images[i]), np.abs(np.amin(images[i]))])
        mask = masks[i]

        input_shape_image = image.shape
        input_shape_mask = mask.shape
        width_size = image.shape[0]
        height_size = image.shape[1]
        width_shift = random.randint(0, width_size - random.randint(np.floor(resize_ratio * width_size), width_size))
        height_shift = random.randint(0, height_size - random.randint(np.floor(resize_ratio * height_size), height_size))
        # adjust brightness, contrast, add noise.  ADD IF NEEDED
        # cropping image
        image = image[width_shift:width_shift + width_size, height_shift:height_shift + height_size]
        mask = mask[width_shift:width_shift + width_size, height_shift:height_shift + height_size]

        # affine transform
        sh = random.random() / 2 - 0.25
        rotate_angle = random.random() * np.pi / 180 * max_angle
        affine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angle)
        image = transform.warp(image, inverse_map=affine_tf, mode='reflect')
        mask = transform.warp(mask, inverse_map=affine_tf, mode='reflect')
        # resize to original size
        image = transform.resize(image, input_shape_image, mode='reflect')
        mask = transform.resize(mask, input_shape_mask, mode='reflect')
        mask = (mask == np.amax(mask))

        images_out.append(image)
        masks_out.append(mask)
    images_out = [(sub_image - np.mean(sub_image)) / np.std(sub_image) for sub_image in images_out]
    return images_out, masks_out


def tile_image(input_image, tile_height, tile_width, edge_loss):
    """
    Takes an image and splits into tiles of specified dimension, mirror padding to ensure uniform size
    :param input_image: The original image to be tiled
    :param tile_height: The vertical size of the desired tile
    :param tile_width: The horizontal size of the desired tile
    :param edge_loss: The edge loss due to the network size so image can be padded to the correct size
    :return: A list of tiles in order [[NW], ..., [SW]; ...; [NE], ..., [SE]],
             as well as the size of the original image for future detiling
    """
    input_height, input_width = np.shape(input_image)
    input_height_original, input_width_original = np.shape(input_image)

    # Check if image has an odd size in any dimension and remove it. This allows us to pad evenly on both sides
    if input_height % 2 != 0:
        input_height -= 1
        input_image = input_image[0:input_height, :]
    if input_width % 2 != 0:
        input_width -= 1
        input_image = input_image[:, 0:input_width]

    # pad the image
    num_tiles_height = int(np.ceil(input_height / tile_height))
    pad_height = int(num_tiles_height * tile_height - input_height)
    num_tiles_width = int(np.ceil(input_width / tile_width))
    pad_width = int(num_tiles_width * tile_width - input_width)
    input_padded = np.pad(input_image, (((pad_height + edge_loss)//2, (pad_height + edge_loss)//2),
                                        ((pad_width + edge_loss)//2, (pad_width + edge_loss)//2)), 'reflect')
    # tile the padding
    tiled_image = [input_padded[n*tile_height:(n+1)*tile_height + edge_loss, m*tile_width:(m+1)*tile_width + edge_loss]
                   for m in range(num_tiles_width) for n in range(num_tiles_height)]
    return tiled_image, input_height_original, input_width_original


def detile_image(tiled_image, input_height_original, input_width_original):
    """
    Takes a tiled image and stitches back together into the original, deleting the padding
    :param tiled_image: A list of tiles, in order given by tile_image- [[NW], ..., [SW]; ...; [NE], ..., [SE]]
    :param input_height_original: The vertical size of the original image, before any padding
    :param input_width_original: The horizontal size of the original image, before any padding
    :return: Single restitched image with all extraneous padding cropped
    """
    # determine the size of each tile
    tile_height, tile_width = np.shape(tiled_image[0])

    # determine the number of tiles in each row and column
    num_tiles_height = int(np.ceil(input_height_original / tile_height))
    num_tiles_width = int(np.ceil(input_width_original / tile_width))

    # combine tiles into columns and then combine columns into a full image
    columns = [np.concatenate(tiled_image[column * num_tiles_height:(column + 1) * num_tiles_height])
               for column in range(num_tiles_width)]
    image = np.concatenate(columns, axis=1)

    # find the center and crop to the original image size
    dim = np.shape(image)
    center = [dim[0]/2, dim[1]/2]
    height_start = int(center[0]-input_height_original/2)
    height_end = int(center[0]+input_height_original/2)
    width_start = int(center[1]-input_width_original/2)
    width_end = int(center[1]+input_width_original/2)
    image = image[height_start:height_end, width_start:width_end]

    return image


def import_images_from_files(image_files, mask_files, downscale=None, tile=None, edge_loss=None):
    """
    Imports all images and associated masks. Includes options for downscaling images and tiling.
    :param image_files: List of file names for images
    :param mask_files: List of file names for masks
    :param downscale: Either None for no downscaling or a tuple (y, x) representing the downscaling in each dimension,
                      or a tuple (y, x, t_y, t_x) representing the downscale in each dimension and the number of tiles
                      in each dimension to force by cropping
    :param tile: Either None for no tiling or a tuple (m,n) representing tiling size in each dimension
    :param edge_loss: The calculated value of the edge loss based on the network architecture
    :return: Outputs images with zero mean and unit variance and masks in proper [0,1] grayscale
    """
    # Import masks
    masks = []
    for file in mask_files:
        mask = rgb2gray(io.imread(file))
        if downscale:
            mask = downscale_local_mean(mask, downscale[0:2])
            if len(downscale) == 4:
                [y_size, x_size] = np.shape(mask)
                x_desired = tile[1] * downscale[3]
                if x_size > x_desired:
                    x_crop = x_size - x_desired
                    mask = mask[:, x_crop//2:-x_crop//2]
                y_desired = tile[0] * downscale[2]
                if y_size > y_desired:
                    y_crop = y_size - y_desired
                    mask = mask[y_crop//2:-y_crop//2, :]
        mask = np.int_(mask > 0)
        if tile:
            mask, _, _ = tile_image(mask, tile[0], tile[1], 0)
        masks.append(mask)
    if tile:
        masks = [tile for image in masks for tile in image]
    print('done importing masks')

    # Import images
    images = []
    for file in image_files:
        #image = rgb2gray(plt.imread(file))
        image = imageio.imread(file)
        if downscale:
            image = downscale_local_mean(image, downscale[0:2])
            if len(downscale) == 4:
                [y_size, x_size] = np.shape(image)
                x_desired = tile[1] * downscale[3]
                if x_size > x_desired:
                    x_crop = x_size - x_desired
                    image = image[:, x_crop//2:-x_crop//2]
                y_desired = tile[0] * downscale[2]
                if y_size > y_desired:
                    y_crop = y_size - y_desired
                    image = image[y_crop//2:-y_crop//2, :]
        if tile:
            image, _, _ = tile_image(image, tile[0], tile[1], edge_loss)
        images.append(image)
    if tile:
        images = [tile for image in images for tile in image]
    else:
        images = [pad_images(image, edge_loss // 2) for image in images]
    images = [(image - np.mean(image)) / np.std(image) for image in images]
    print('done importing images')
    return images, masks


def post_process(mask):
    filled = ndi.binary_fill_holes(mask)
    label_objects, nb_labels = ndi.label(filled)
    sizes = np.bincount(label_objects.ravel())
    sorted_sizes = np.sort(sizes)
    mask_sizes = sizes > sorted_sizes[-2] - 1
    mask_sizes[0] = 0
    mask_cleaned = mask_sizes[label_objects]
    return mask_cleaned


def dice_coefficient(labels, prediction):
    intersection = np.sum(labels * prediction)
    dice_coef = (2. * intersection) / (np.sum(labels) + np.sum(prediction))
    return 1.0 - dice_coef
