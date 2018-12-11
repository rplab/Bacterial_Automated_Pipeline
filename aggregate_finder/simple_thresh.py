


import numpy as np
from skimage.measure import label, regionprops
from matplotlib import patches
from local_functions import *
from skimage.filters import threshold_otsu
from scipy.misc import imsave
import operator
from time import time


directory = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/ae1/biogeog_1_2/scans/region_3/'
save_directory = '/media/parthasarathy/Stephen Dedalus/bac cluster_segment_test/ae1_f1_2_r3/'

time_init = time()
files = glob(directory + '/*.tif')
files = [file for file in files if 'mask' not in file]
sort_nicely(files)
images = []
for file in files:
    image = plt.imread(file)
    images.append(image)



test = np.array(images).flatten()
thresh = np.mean(test) + 8 * np.std(test)
for i in range(len(images)):
    imsave(save_directory + str(i) + '.tif', np.concatenate((images[i], (images[i] > thresh)*images[i]), axis=1))




