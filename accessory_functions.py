

import numpy as np
from skimage import exposure
from skimage.feature import blob_dog
from skimage.measure import block_reduce
from skimage.morphology import binary_erosion
from skimage.filters import threshold_otsu
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from time import time
from scipy import ndimage
import pickle
from matplotlib import pyplot as plt
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


def dist(x1, y1, x2y2):
    x2 = x2y2[0]
    y2 = x2y2[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def blobTheBuilder(files, scale, min_sig, max_sig, thrsh):
    global blobs
    global start_time
    global plots
    plots = []
    start_time = time()
    blobs = []
    print('starting loop')
    t_read = 0
    t_reduce = 0
    t_resize = 0
    t_blob = 0
    t_append = 0
    for name in files:
        t0 = time()
        image = ndimage.imread(name, flatten=True)
        t1 = time()
        t_read += t1-t0
        image = block_reduce(image, block_size=(scale, scale), func=np.mean)
        plots.append(image.tolist())
        t2 = time()
        t_reduce += t2-t1
        image = (image - np.min(image))/np.max(image)
        t3 = time()
        t_resize += t3-t2
        tempblobs = blob_dog(image, max_sigma=max_sig, min_sigma=min_sig, threshold=thrsh, overlap=0).tolist()  # I am concerned that I want to be
        # varying the threshold based on the intensity of the z-plane image somehow.
        for tempblob in tempblobs: # Is this done too early? What am I doing?
            tempblob.append(0)
            tempblob.append(0)
        t4 = time()
        t_blob += t4-t3
        if tempblobs == []:
            blobs.append([[]])
        else:
            blobs.append(tempblobs)
        t5 = time()
        t_append += t5-t4
        # print(str(round(time() - t0, 1)) + 'seconds')
        print(name)
    print('t_read = ' + str(round(t_read, 1)))
    print('t_reduce = ' + str(round(t_reduce, 1)))
    print('t_resize = ' + str(round(t_resize, 1)))
    print('t_blob = ' + str(round(t_blob, 1)))
    print('t_append = ' + str(round(t_append, 1)))
    return(blobs)


def segmentation_mask(plots1, wdth, thresh2):
    plots_out = [[] for el in range(len(plots1))]
    plots2 = [[] for el in range(len(plots1))]
    print('building mask...')
    for i in range(len(plots1)):
        image = plots1[i]
        image = (image - np.min(image))/np.max(image)
        plots2[i] = exposure.equalize_hist(np.array(image))
    for i in range(len(plots2)):
        if i < int(wdth/2):
            image = np.mean(plots2[0: wdth], axis=0)
        elif i > int(len(plots2) - wdth/2):
            image = np.mean(plots2[-wdth: -1], axis=0)
        else:
            image = np.mean(plots2[i-int(wdth/2):i+int(wdth/2)], axis=0)
        binary = image > thresh2
        plots_out[i] = binary
    return plots_out


def trim_segmented(blobs, wdth, thresh2):
    global trim_time
    plots1 = segmentation_mask(plots, wdth, thresh2)
    trim_time = time()
    print('done building the mask')
    for z in range(len(blobs)):
        rem = []
        for blob in blobs[z]:
            if plots1[z][int(blob[0])][int(blob[1])] is False and blob != []:  # need to check if z, x, y is correct ordering
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


def trim_consecutively(blobs, adjSize):
    for z in range(len(blobs)):
        for n in range(len(blobs[z])):
            if blobs[z][n][2] == 0:
                break
            else:
                blobs[z][n][2] = 1
                contains = 'True'
                zz = z + 1
                testlocation = blobs[z][n][0:2]
                while contains == 'True' and zz < len(blobs):
                    if blobs[zz] == []: #  check for empty zz
                        break
                    for blob in blobs[zz]:
                        if dist(blob[0], blob[1], testlocation) < adjSize:
                            blobs[z][n][2] += 1
                            testlocation = blob[0:2]
                            # x-end
                            blobs[z][n][3] = testlocation[0]
                            # x-end
                            blobs[z][n][4] = testlocation[1]

                            blobs[zz].remove(blob)
                            zz += 1
                            contains = 'True'
                            break
                        else:
                            contains = 'False'
    return blobs


def trim_toofewtoomany(blobs, tooFew, tooMany, xpixlength, ypixlength, cubeLength=30):
    for z in range(len(blobs)):
        rem = []    # note, removing while looping skips every other entry to be removed
        for blob in blobs[z]:
            if blob[2] < tooFew or blob[2] > tooMany:
                rem.append(blob)
            # the following makes sure blobs aren't on x-y edge of image
            elif blob[0] < cubeLength or blob[1] < cubeLength:
                rem.append(blob)
            elif blob[0] > xpixlength - cubeLength:
                rem.append(blob)
            elif blob[1] > ypixlength - cubeLength:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


def cubeExtractor(files, blibs, cubeLength=30, zLength = 10):  # Maybe want sliding cube?
    z = 0
    cubes = [[] for el in blibs]
    for name in files:
        z += 1
        image = ndimage.imread(name, flatten=True)  # CHANGE TO EXTRACT FROM PLOTS
        # TO SCALE IMAGES

        for el in range(len(blibs)):
            if blibs[el][2] > len(blobs) - int(zLength/2) and z > len(blobs) - zLength:
                xstart = int(blibs[el][0] - cubeLength / 2)
                ystart = int(blibs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
            elif blibs[el][2] > z + int(zLength/2):
                break
            elif blibs[el][2] <= int(zLength/2) and z <= zLength:
                xstart = int(blibs[el][0] - cubeLength / 2)
                ystart = int(blibs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
            elif blibs[el][2] > z - int(zLength/2):
                xstart = int(blibs[el][0] - cubeLength/2)
                ystart = int(blibs[el][1] - cubeLength/2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
    print('total time up to cube extraction = ' + str(round(time() - start_time, 1)))
    return cubes


def mask_blob_trim(files, scale=4, min_sig=2, max_sig=2, thrsh=0.02, wdth=30, thresh2=0.8,
                   adjSize=2, tooFew=2, tooMany=15):
    pix_dimage = ndimage.imread(files[0], flatten=True)
    ypixlength = len(pix_dimage[0])
    xpixlength = len(pix_dimage)
    blobs = blobTheBuilder(files, scale, min_sig, max_sig, thrsh)
    blobs = trim_segmented(blobs, wdth, thresh2)
    blobs = trim_consecutively(blobs, adjSize)
    blobs = trim_toofewtoomany(blobs, tooFew, tooMany, xpixlength, ypixlength)
    print('Total time to trim blobs = ' + str(round(time() - trim_time, 1)))
    #  blibs is one-d list of (x,y,z, bac) for detected blobs
    blibs = [[blobs[i][n][0] * scale + (blobs[i][n][3] - blobs[i][n][0]) / 2 * scale,
              blobs[i][n][1] * scale + (blobs[i][n][4] - blobs[i][n][1]) / 2 * scale,
              int(i + blobs[i][n][2] / 2)] for i in range(len(blobs)) for n in range(len(blobs[i]))]
    blibs = sorted(blibs, key=lambda x: x[2])
    cubes = cubeExtractor(files, blibs)
    return cubes, blibs
