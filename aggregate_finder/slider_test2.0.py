

from unet.data_processing import *

from time import time
from skimage.filters import laplace, sobel, scharr
from skimage.morphology import binary_closing, binary_opening, disk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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
    ax.cla()
    ax.matshow(images[n], cmap="Blues_r")
    threshold[int(n/5)] = sthresh.val
    mask = mask_func(images[n], threshold[int(n/5)], mip)
    masked = np.ma.array(images[n], mask=mask)
    masked_ax = ax.imshow(masked, alpha=0.5, cmap='Oranges_r')


def load_data(directory_loc):
    files = glob(directory + '/*.tif')
    files = [file for file in files if 'mask' not in file]
    sort_nicely(files)
    images = []
    for file in files:
        images.append(plt.imread(file))
        print(file)
    print('done loading images')
    return images


drive = drive_loc('Stephen Dedalus')
directory = drive + '/zebrafish_image_scans/bac_types/ae1/biogeog_1_3/scans/region_1/'
mip = plt.imread(directory.split('/scans')[0] + '/MIPS/' + 'region_' + directory.split('region_')[1][0] + '.tif')
# directory  ='/media/parthasarathy/Stephen Dedalus/Multi-Species/Multi-Species/germ free/di/11_21_2018/AE-RFP__EN-GFP/' \
#             'fish6/fish1/Scans/scan_1/region_1/568nm'
# mip = plt.imread(directory.split('fish1')[0] + 'MIPS/' + 'region_' + directory.split('region_')[1][0] + '.tif')
mip = np.abs(mip - np.amax(mip))/ np.amax(mip)
images = load_data(directory)



n = 0
selem = disk(6)
threshold = [0.003 for i in range(1, len(images), 5)]
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
fig.suptitle('z = ' + str(n))
axcolor = 'lightgoldenrodyellow'
axthresh = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
sthresh = Slider(axthresh, 'threshold', 0.0005, 0.005, valinit=threshold[int(n/5)], valfmt='%.4f')
sthresh.on_changed(update)
plotInit(n)




