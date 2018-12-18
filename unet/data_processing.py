


import numpy as np
from glob import glob
import re
from scipy import ndimage
from sklearn.model_selection import train_test_split
from skimage.transform import downscale_local_mean
from skimage import transform
import random
from pathlib import Path



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


###  DATA PROCESSING  --
def data_augment(images, masks, angle=5, resize_rate=0.9):  # Adjust params. Combine with data read-in in training?
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


def read_in_images(directory_loc, label_string='_gutmask', size2=256, size1=200, test_size=0.0, import_length=-1,
                   downsample=2):
    files = glob(directory_loc + '/*.tif', recursive=True)
    sort_nicely(files)
    mask_files = [item for item in files if label_string in item][:import_length]
    data_files = [re.sub(label_string, '', item) for item in mask_files][:import_length]
    if downsample == 1:
        masks = [ndimage.imread(file) for file in mask_files]
        data = [ndimage.imread(file) for file in data_files]
    else:
        masks = [downscale_local_mean(ndimage.imread(file), (2, 2)) for file in mask_files]
        data = [downscale_local_mean(ndimage.imread(file), (2, 2)) for file in data_files]
    print('data length: ' + str(len(data)))
    print('data length: ' + str(len(data)))
    print('done reading in previous masks and data')
    tiled_masks = []
    for i in range(len(masks)):
        temp_masks = tile_image(masks[i], size2=size1, size1=size1)
        for sub_mask in temp_masks:
            sub_mask = np.resize(sub_mask, (size1, size1))
            sub_mask = sub_mask/np.max([np.amax(sub_mask), 1])
            tiled_masks.append(sub_mask)
    data = [(sub_image - np.mean(sub_image)) / np.std(sub_image) for sub_image in data]
    print('done reading in new masks')
    tiled_data = []
    for i in range(len(data)):
        temp_data = tile_image(data[i], size2=size2, size1=size1)
        for sub_image in temp_data:
            sub_image = np.resize(sub_image, (size2, size2))
            tiled_data.append(sub_image)
    print('total data: ' + str(len(data)))
    return train_test_split(tiled_data, tiled_masks, test_size=test_size)


def drive_loc(drive_name):
    if drive_name == 'Bast':
        if str(Path.home()).split('/')[-1] == 'teddy':
            drive_name = 'Bast1'
        else:
            drive_name = 'Bast'
    drive = '/media/' + str(Path.home()).split('/')[-1] + '/' + drive_name
    return drive


def detile_1(image, input_tiles, size1=372):
    size = np.shape(input_tiles[0])[0]
    x1 = [n * size for n in range(np.shape(image)[0] // size + 1)]
    x2 = x1[1:] + [-1]
    y1 = [n * size for n in range(np.shape(image)[1] // size + 1)]
    y2 = y1[1:] + [-1]
    stitched_image = np.zeros(np.shape(image))
    iter = 0
    for x in range(len(x1)):
        for y in range(len(y1)):
            shape = np.shape(stitched_image[x1[x]:x2[x], y1[y]:y2[y]])
            x_size = 0
            y_size = 0
            if x == len(x1) - 1:
                x_size = size//2 - shape[0]//2
            if y == len(y1) - 1:
                y_size = size//2 - shape[1]//2
            stitched_image[x1[x]:x2[x], y1[y]:y2[y]] = \
                np.array(input_tiles[iter])[x_size: x_size + shape[0], y_size: y_size + shape[1]]
            iter += 1
    return stitched_image




