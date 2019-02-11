



from scipy.misc import imsave
from unet.data_processing import *
from matplotlib import pyplot as plt
from time import time
from skimage.filters import rank
from skimage.filters import laplace, sobel, scharr
from skimage.morphology import disk, closing, square


drive = drive_loc('Stephen Dedalus')
directory = drive + '/zebrafish_image_scans/bac_types/ae1/biogeog_1_3/scans/region_1/'
save_directory = drive + '/bac cluster_segment_test/ae1_f1_3_r2/'

time_init = time()
files = glob(directory + '/*.tif')
files = [file for file in files if 'mask' not in file]
sort_nicely(files)
images = []
for file in files:
    image = plt.imread(file)
    images.append(image)
print('done loading images')


def key_z_plots(e):
    global n
    if e.key == "right":
        n = n + 1
    elif e.key == "left":
        n = n - 1
    else:
        return
    plotInit(n)

def plotInit(n):
    fig.canvas.mpl_connect('key_press_event', key_z_plots)
    ax1.imshow(images[n])
    # ax2.imshow((images[n] > thresh) * images[n])
    ax2.imshow(closing(closing((scharr(images[n]) > 0.003), square(5))) * images[n])

n = 50
thresh = 5000
fig, (ax1, ax2) = plt.subplots(1, 2)
plotInit(n)
selem = disk(5)
plt.imshow(closing((sobel(images[n]) > 0.003), selem) * images[n])
plt.imshow(sobel(images[n]))
