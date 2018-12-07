


import numpy as np
from skimage import exposure
from scipy import ndimage
from scipy.misc import imsave
from matplotlib import pyplot as plt
from skimage.measure import label
import re
from glob import glob
from skimage.measure import block_reduce


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


def segmentation_mask(images_in, width=10, thresh=0.7):
    plots_out = [[] for el in range(len(images_in))]
    plots2 = [[] for el in range(len(images_in))]
    print('building mask...')
    for i in range(len(images_in)):
        image = images_in[i]
        image = (image - np.min(image))/np.max(image)
        plots2[i] = exposure.equalize_hist(np.array(image))
    for i in range(len(plots2)):
        if i < int(width/2):
            image = np.mean(plots2[0: width], axis=0)
        elif i > int(len(plots2) - width/2):
            image = np.mean(plots2[-width: -1], axis=0)
        else:
            image = np.mean(plots2[i-int(width / 2):i + int(width / 2)], axis=0)
        binary = image > thresh
        binary = getLargestCC(binary)
        plots_out[i] = binary
    return plots_out


def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC

directory_loc = '/media/parthasarathy/Bast/Teddy/emptyGut_6_28_16_secondFish/fish3/fish1/Scans/scan_1/region_3/488nm'
files = glob(directory_loc + '/*.png')
# files = [file for file in files if 'gutmask' not in file]
sort_nicely(files)

images = []
for file in files:
    image = ndimage.imread(file, flatten=True)
    image = block_reduce(image, block_size=(4, 4))
    images.append(image)
print(np.shape(images))

test = segmentation_mask(images)
segmentation = np.array(test)
test = getLargestCC(np.array(test))

test = ~test

save_loc = '/media/parthasarathy/Bast/UNET_Projects/intestinal_outlining/plots and figs/Examples/empty_hist_segment_fullscan'
for test_int in range(len(test)):
    imsave(save_loc + '/' + str(test_int) + '.png', test[test_int]*images[test_int])


