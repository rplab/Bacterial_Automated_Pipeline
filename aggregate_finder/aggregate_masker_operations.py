
from matplotlib import pyplot as plt
import glob
import os
import re
import numpy as np
from skimage.morphology import binary_closing, binary_opening, binary_erosion, disk
from skimage import morphology
import numpy as np
from skimage import restoration, segmentation
from skimage.segmentation import join_segmentations
from scipy import ndimage as ndi
from skimage.restoration import denoise_wavelet, cycle_spin, denoise_bilateral
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure


def sort_nicely(l):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def find_region_and_color(direc):
    region_number_and_color = np.unique([file.split('region_')[1] for file in direc])[0]
    get_color = region_number_and_color[2:7]
    get_region = region_number_and_color[0]
    return str(get_region), str(get_color)

def save_mask_loc(direc):
    scan_loc = np.unique([file.split('region_')[0] for file in direc])[0]
    return scan_loc

def load_gutmask(direc, first_image):
    first_image_location = direc[0]
    scan_loc = glob.glob(np.unique([first_image_location.split('Scans')[0]])[0] + '/**/*/**')
    region, color = find_region_and_color(direc)
    gut_mask_direc = [file for file in scan_loc if 'scan_1_region_'+str(region) + '_'+ str(color) +'_gutmask' in file]
    gut_mask_3D = np.load(gut_mask_direc[0])['gutmask']
    nogutmask = np.logical_not(np.zeros(np.shape([first_image]), dtype=bool))

    #### empty gutmask in case user wants to remove gutmask before segmentation ### +#

    if np.shape(gut_mask_3D[0])[0] != np.shape((first_image))[0]:
        print('Shape of gut mask and image do not match! Clipping gut mask')
        gut_mask_3D = [np.delete(gut_mask_3D[i], len(gut_mask_3D[0]) - 1, axis=0) for i in range(len(gut_mask_3D))]
    return gut_mask_3D, nogutmask

def denoising_filter_image_stack(direc):

    #### initialize empty image list to populate images###

    full_image_3D = [0] * len(direc)

    for z in range(5):#len(direc)):
        img = plt.imread(direc[z])
        img = restoration.denoise_bilateral(img, multichannel=False)
        full_image_3D[z] = (img - img.min()) / (img.max() - img.min())
        print('Completed denoising image ' + str(z) + ' of ' + str(len(full_image_3D) - 1))

    return full_image_3D