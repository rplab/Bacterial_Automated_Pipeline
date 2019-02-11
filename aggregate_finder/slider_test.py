
from matplotlib.widgets import Slider
from unet.data_processing import *
from matplotlib import pyplot as plt
from time import time
from skimage.filters import laplace, sobel, scharr
from skimage.morphology import binary_closing, binary_opening, disk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def mask_func(image, threshold):
    return binary_closing(binary_closing((scharr(image) > threshold), selem), selem)


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
    threshold = sthresh.val
    mask = mask_func(images[n], threshold)
    print('here2')
    masked.mask = mask
    masked_ax.set_data(masked)
    fig.canvas.draw_idle()


def plotInit(n):
    global masked_ax
    global masked
    global sthresh
    global threshold
    fig.canvas.mpl_connect('key_press_event', key_z_plots)
    ax.cla()
    ax.matshow(images[n], cmap="Blues_r")
    threshold = sthresh.val
    mask = mask_func(images[n], threshold)
    masked = np.ma.array(images[n], mask=mask)
    masked_ax = ax.imshow(masked, alpha=0.5, cmap=plt.cm.gray)


drive = drive_loc('Stephen Dedalus')
directory = drive + '/zebrafish_image_scans/bac_types/ae1/biogeog_1_3/scans/region_1/'
save_directory = drive + '/bac cluster_segment_test/ae1_f1_3_r2/'

time_init = time()
files = glob(directory + '/*.tif')
files = [file for file in files if 'mask' not in file]
sort_nicely(files)
images = []
for file in files:
    images.append(plt.imread(file))
print('done loading images')




n = 0
selem = disk(6)
threshold = 0.003
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
fig.suptitle('z = ' + str(n))
axcolor = 'lightgoldenrodyellow'
axthresh = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
sthresh = Slider(axthresh, 'threshold', 0.0005, 0.005, valinit=threshold, valfmt='%.4f')
sthresh.on_changed(update)
plotInit(n)





