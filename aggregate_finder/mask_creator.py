

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import scharr
from skimage.morphology import binary_closing, disk
from glob import glob
from accessory_functions import sort_nicely


def load_data_create_mask(directory_loc):
    sub_directory = directory_loc.split('z_thresh')[0] + 'fish1/region_' + directory_loc.split('region_')[1][0] #+ '/' + directory_loc.split('.npz')[0][-5:]
    mip = plt.imread(sub_directory.split('/fish1')[0] + '/MIPS/' + 'region_' + sub_directory.split('region_')[1][0] +
                     '.tif')
    mip = np.abs(mip - np.amax(mip)) / np.amax(mip)
    thresh_profile = np.load(directory_loc)['threshold']
    files = glob(sub_directory + '/' + directory_loc.split('.npz')[0][-5:] + '/*.tif')
    sort_nicely(files)
    thresh_profile = np.pad(thresh_profile, (0, len(files) - len(thresh_profile)), 'constant',
                            constant_values=(thresh_profile[0], thresh_profile[-1]))
    print([len(thresh_profile), len(files)])
    masks = []
    for n in range(len(files)):
        image = plt.imread(files[n])
        masks.append(mask_func(image, thresh_profile[n], mip))
        print(files[n])
    print('done loading images')
    np.savez(directory_loc.split('z_thresh')[0] + 'mask_region_' + directory_loc.split('region_')[1], masks)
    print('saved mask')


def mask_func(image, threshold, mip):
    return binary_closing(binary_closing((scharr(image, mask=mip) > threshold), selem), selem)


selem = disk(6)
directory = '/media/rplab/Dagobah/deepika/Aggregate_masks/diassociation_time_series/'
files = glob(directory + '**/*.npz', recursive=True)
files = [file for file in files if 'mask_' not in file]
# files = [file for file in files if 'mono/en' in file]
sort_nicely(files)
directory_loc = files[0]
for directory_loc in files:
    load_data_create_mask(directory_loc)
