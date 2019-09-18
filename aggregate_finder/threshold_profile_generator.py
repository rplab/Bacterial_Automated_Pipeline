

from unet.data_processing import *
from skimage.filters import laplace, sobel, scharr
from skimage.morphology import binary_closing, binary_opening, disk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from glob import glob
import re


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


def mask_func(image, threshold, mip):
    return binary_closing(binary_closing((scharr(image, mask=mip) > threshold), selem), selem)


def key_z_plots(e):
    global n
    if e.key == "right":
        n = n + 5
    elif e.key == "left":
        n = n - 5
    else:
        return
    fig.suptitle('z = ' + str(n))
    plotInit(n)


def key_gut_range(e):
    global n
    if e.key == "b":
        gut_range[0] = n
        print(gut_range)
    elif e.key == "e":
        gut_range[-1] = n
        print(gut_range)
    else:
        return


def key_save_thresh(e):
    global threshold
    if e.key == 'escape':
        plt.close()
        threshold_out = [i for i in threshold for j in range(5)][:len(images)]
        name = directory_loc.split('fish1')[0] + 'z_thresh_vals_region_' + directory_loc.split('region_')[1].replace('/', '_')
        np.savez_compressed(name, threshold=threshold_out, gut_range=gut_range)
    else:
        return


# def key_mask_off(e):
#     global mask_incr
#     if e.key == 'enter':
#         mask_incr *= -1
#         plotInit(n)
#     else:
#         return


def update(val):
    threshold[int(n/5)] = sthresh.val
    mask = mask_func(images[n], threshold[int(n/5)], mip)
    masked.mask = mask
    masked_ax.set_data(masked)
    fig.canvas.draw_idle()


def plotInit(n):
    global masked_ax
    global mip
    global masked
    global sthresh
    global threshold
    fig.canvas.mpl_connect('key_press_event', key_z_plots)
    fig.canvas.mpl_connect('key_press_event', key_gut_range)
    fig.canvas.mpl_connect('key_press_event', key_save_thresh)
    # fig.canvas.mpl_connect('key_press_event', key_mask_off)
    ax.cla()
    ax.matshow(images[n], cmap="Blues_r")
    if mask_incr==1:
        threshold[int(n/5)] = sthresh.val
        mask = mask_func(images[n], threshold[int(n/5)], mip)
        masked = np.ma.array(images[n], mask=mask)
        masked_ax = ax.imshow(masked[10:], alpha=0.5, cmap='Oranges_r')


def load_data(directory_loc):
    files = glob(directory_loc + '/*.tif')
    # files = [file for file in files if 'mask' not in file]
    sort_nicely(files)
    images = []
    for file in files:
        images.append(plt.imread(file))
        print(file)
    print('done loading images')
    return images


# drive = drive_loc('Stephen Dedalus')
# directory = drive + '/zebrafish_image_scans/bac_types/ae1/biogeog_1_3/scans/region_1/'
directory_loc = '/media/rplab/Dagobah/deepika/Aggregate_masks/diassociation_time_series/7_24_scan_6/Scans/fish1/region_2/568nm'
mip = plt.imread(directory_loc.split('/fish1')[0] + '/MIPS/' + 'region_' + directory_loc.split('region_')[1][0] + '.tif')
# directory  ='/media/parthasarathy/Stephen Dedalus/Multi-Species/Multi-Species/germ free/di/11_21_2018/AE-RFP__EN-GFP/' \
#             'fish6/fish1/Scans/scan_1/region_1/568nm'
# mip = plt.imread(directory.split('fish1')[0] + 'MIPS/' + 'region_' + directory.split('region_')[1][0] + '.tif')
mip = np.abs(mip - np.amax(mip)) / np.amax(mip)
images = load_data(directory_loc)
threshold = [0.003 for i in range(1, len(images), 5)]
gut_range = [0, -1]



# loaded = np.load(directory.split('region')[0] + 'z_thresh_vals_region_' + directory.split('region_')[1].replace('/', '_') + '.npz')
# threshold = loaded['threshold']
n = 0
selem = disk(6)
mask_incr = 1
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
fig.suptitle('z = ' + str(n))
axcolor = 'lightgoldenrodyellow'
axthresh = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
sthresh = Slider(axthresh, 'threshold', 0.0001, 0.005, valinit=threshold[int(n/5)], valfmt='%.4f')
sthresh.on_changed(update)
plotInit(n)



plt.figure()
plt.plot(threshold)




