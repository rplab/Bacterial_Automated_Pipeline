

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from pathlib import Path
from glob import glob
import re
from scipy import ndimage
from sklearn.model_selection import train_test_split
from skimage.exposure import equalize_hist
from skimage.transform import downscale_local_mean


def drive_loc(drive_name):
    if drive_name == 'Bast':
        if str(Path.home()).split('/')[-1] == 'teddy':
            drive_name = 'Bast1'
        else:
            drive_name = 'Bast'
    drive = '/media/' + str(Path.home()).split('/')[-1] + '/' + drive_name
    return drive


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


def read_in_images(directory_loc, label_string='_gutmask', test_size=0.0, resize_to=512):
    files = glob(directory_loc + '/*.tif', recursive=True)
    files.extend(glob('*.png'))
    files = [file for file in files if any(['region_1' in file, 'region1' in file])]
    sort_nicely(files)
    mask_files = [item for item in files if label_string in item]
    data_files = [re.sub(label_string, '', item) for item in mask_files]
    data = []
    for file in data_files:
        image = rgb2gray(plt.imread(file))
        image_resized = resize(image[10:], (resize_to, resize_to), anti_aliasing=True)
        image_resized = (image_resized - np.mean(image_resized))/np.amax(image_resized)
        data.append(image_resized)
    masks = []
    for file in mask_files:
        mask = plt.imread(file)
        mask_resized = resize(mask[10:], (resize_to, resize_to), anti_aliasing=True)
        mask_resized_normed = mask_resized/np.max([np.amax(mask_resized), 1])
        mask_resized_normed = np.int_(mask_resized_normed == 0)
        masks.append(mask_resized_normed)
    print('done reading in previous masks and data')
    print('total data: ' + str(len(data)))
    print('total masks: ' + str(len(masks)))
    return train_test_split(data, masks, test_size=test_size)


def pad_images(images, pad_to):
    images_out = [np.pad(image, ((pad_to, pad_to), (pad_to, pad_to)), mode='reflect') for image in images]
    return images_out
