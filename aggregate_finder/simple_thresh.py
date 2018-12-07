

from matplotlib import gridspec
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import patches
from local_functions import *
from skimage.filters import threshold_otsu
from scipy.misc import imsave
import operator
from time import time



def initial_thresholding(images, thresh, min_box_allowed1):
    binarized = [binary_erosion(closing(image, square(3)) > thresh) for image in images]
    labeled = label(np.array(binarized))
    props = regionprops(labeled)
    print(len(props))
    neuts = []
    for i in range(len(props)):
        if all(np.array(np.shape(props[i].image)) > min_box_allowed1):
            neuts.append(props[i])
    return neuts


def shrinking_boxes(images, neuts, min_box_allowed2, min_area_allowed, min_area_to_shrink):
    new_mask = np.zeros(np.shape(images))
    for i in neuts:
        (z_start, y_start, x_start, z_end, y_end, x_end) = i.bbox
        neut_sub_image = np.array(images)[z_start:z_end, y_start:y_end, x_start:x_end]
        thresh_sub_image = threshold_otsu(neut_sub_image)
        if i.area < min_area_to_shrink:
            neut_sub_mask = np.array(neut_sub_image > thresh_sub_image)
        else:
            neut_sub_mask = np.array(binary_erosion(neut_sub_image > thresh_sub_image))
        props_sub_images = regionprops(label(neut_sub_mask))
        for el in props_sub_images:
            (z_start, y_start, x_start) = tuple(map(operator.add, i.bbox[:3], el.bbox[:3]))
            (z_end, y_end, x_end) = tuple(map(operator.add, i.bbox[:3], el.bbox[-3:]))
            new_mask[z_start:z_end, y_start:y_end, x_start:x_end] = el.image
    labeled = label(np.array(new_mask))
    props = regionprops(labeled)
    neuts = []
    for i in range(len(props)):
        if all(np.array(np.shape(props[i].image)) > min_box_allowed2) and props[i].area > min_area_allowed:
            neuts.append(props[i])
    return neuts


def mean_sem(dataset):
    return [np.round(np.mean(dataset), 2), np.round(np.std(dataset)/len(dataset), 2)]


directory = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/ae1/biogeog_1_2/scans/region_3/'
save_directory = '/media/parthasarathy/Stephen Dedalus/bac cluster_segment_test/ae1_f1_2_r3/'
# Median Params = [849, 3, 3, 16, 7627]
thresh = 849
n = 3
min_box_allowed1 = [n, 2*n, 2*n]
m = 3
min_box_allowed2 = [m, 2*m, 2*m]
min_area_allowed = 16
min_area_to_shrink = 7627
shrink_iterations = 3


total_counts = [[], [], [], []]
xyzs = [[], [], [], []]
i = 0
time_init = time()
files = glob(directory + '/*.tif')
files = [file for file in files if 'mask' not in file]
sort_nicely(files)
images = []
for file in files:
    image = plt.imread(file)
    images.append(image)



test = np.array(images).flatten()
thresh = np.mean(test) + 8*np.std(test)
image = images[85]
plt.imshow(image)
plt.figure()
np.shape(image)
plt.imshow((image > thresh)*image)
for i in range(len(images)):
    imsave(save_directory + str(i) + '.tif', np.concatenate((images[i], (images[i] > thresh)*images[i]), axis=1))

print([np.amin(images), np.amax(images)])
### INITIAL THRESHOLDING AT FIXED THRESH
neuts = initial_thresholding(images, thresh, min_box_allowed1)
print(len(neuts))
### ITERATIVE THRESHOLDING USING SHRINKING BOXES
for j in range(shrink_iterations):
    neuts = shrinking_boxes(images, neuts, min_box_allowed2, min_area_allowed, min_area_to_shrink)
    print(str(len(neuts)) + ' neutrophil')




# neut_counts.append(len(neuts))
# region_centroids.append([i.centroid for i in neuts])
# print('time per region = ' + str(np.round((time() - time_init)/60, 2)))
#
# new_mask = np.zeros(np.shape(images))
# fig = plt.figure(figsize=(10, 10))
# gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.imshow(np.max(images, axis=0))
# # ax1.set_title(str(increment) + '__' + specific_files[0].split('/')[-2][-8:])
# ax2 = fig.add_subplot(gs[1, 0])
# ax2.imshow(np.max(images, axis=1))
# ax3 = fig.add_subplot(gs[2, 0])
# ax3.imshow(np.max(images, axis=2))
# for i in neuts:
#     (z_size, y_size, x_size) = np.shape(i.image)
#     (z_start, y_start, x_start, z_end, y_end, x_end) = i.bbox
#     r1 = patches.Rectangle((x_start, y_start), x_size, y_size, color='red', linewidth=1,
#                           fill=False)
#     r2 = patches.Rectangle((x_start, z_start), x_size, z_size, color='red', linewidth=1,
#                           fill=False)
#     r3 = patches.Rectangle((y_start, z_start), y_size, z_size, color='red', linewidth=1,
#                           fill=False)
#     ax1.add_patch(r1)
#     ax2.add_patch(r2)
#     ax3.add_patch(r3)



