



from matplotlib import pyplot as plt
from matplotlib import gridspec, patches
import numpy as np
import pickle
from skimage.feature import blob_dog
from skimage.measure import block_reduce
from skimage import exposure
from time import time
from scipy import ndimage
import glob
import os.path
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


def preamble():
    global run
    global fileNames
    global table_name
    global fileLoc
    global bacteria_type
    global usrname
    global xpixlength
    global ypixlength
    global output_file
    global cubes
    global ROI_locs
    folder_location = input('copy paste (CTRL+SHFT+v) the file location of your first image please:  ')
    print()
    bacteria_type = input('What type of bacteria are you identifying?  ')
    fileLoc = folder_location
    fileNames = glob.glob(fileLoc + '/*.tif')
    fileNames.extend(glob.glob(fileLoc + '/*.png'))
    fileNames = [file for file in fileNames if 'gutmask' not in file]
    sort_nicely(fileNames)
    pix_dimage = ndimage.imread(fileNames[0], flatten=True)
    ypixlength = len(pix_dimage[0])
    xpixlength = len(pix_dimage)


def dist(x1, y1, list):
    x2 = list[0]
    y2 = list[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def blobTheBuilder(start, stop, scale, min_sig=0.3, max_sig=20, thrsh=0.02):
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
    if bacteria_type == 'z20':
        min_sig = 0.3
        max_sig = 20
        thrsh = 0.02
    elif bacteria_type == 'en':
        min_sig = 0.05
        max_sig = 4
        thrsh = 0.02
    elif bacteria_type == 'ps':
        min_sig = 2
        max_sig = 10
        thrsh = 0.02
    elif bacteria_type == 'ao1':
        min_sig = 0.05
        max_sig = 4
        thrsh = 0.03
    else:
        print('No preset size for this bacteria -- Using input values or defaults')

    for name in fileNames[start:stop]:
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
        for tempblob in tempblobs:
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


def trim_segmented(blobs, wdth=30, thresh2=0.7):
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
#                                Loop through blobs trimming consecutive blobs
#                        !!I am not going back to original scale till after trimming!!

# HERE I NEED TO SAVE FIRST AND LAST X-Y LOCATION FOR EACH BLOB TO FIND X-Y CENTER FOR OUTPUT

def trim_consecutively(blobs, adjSize=2):
    for z in range(len(blobs)):
        for n in range(len(blobs[z])):
            if blobs[z][n][2] == 0:
                break
            else:
                blobs[z][n][2] = 1
                contains = 'True'
                zz = z + 1
                testlocation = blobs[z][n][0:2]
                # firstlocation = testlocation
                # blobx = 10000
                # bloby = 10000
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
                # z_stretch = dist(blobx, bloby, firstlocation)
                # blobs[z][n].append(z_stretch)
    return blobs


#                            trim when blob only in one or two planes
def trim_toofewtoomany(blobs, tooFew=2, tooMany=15):
    for z in range(len(blobs)):
        rem = []    # note, removing while looping skips every other entry to be removed
        for blob in blobs[z]:
            if blob[2] < tooFew or blob[2] > tooMany:
                rem.append(blob)
            # the follwing makes sure blobs aren't on x-y edge of image
            elif blob[0] < cubeLength or blob[1] < cubeLength:
                rem.append(blob)
            elif blob[0] > xpixlength - cubeLength:
                rem.append(blob)
            elif blob[1] > ypixlength - cubeLength:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


def cubeExtractor():  # Maybe want sliding cube?
    z = 0
    cubes = [[] for el in ROI_locs]
    for name in fileNames[start:stop]:
        z += 1
        image = ndimage.imread(name, flatten=True)  # CHANGE TO EXTRACT FROM PLOTS
        for el in range(len(ROI_locs)):
            if ROI_locs[el][2] > len(blobs) - int(zLength / 2) and z > len(blobs) - zLength:
                xstart = int(ROI_locs[el][0] - cubeLength / 2)
                ystart = int(ROI_locs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
            elif ROI_locs[el][2] > z + int(zLength / 2):
                break
            elif ROI_locs[el][2] <= int(zLength / 2) and z <= zLength:
                xstart = int(ROI_locs[el][0] - cubeLength / 2)
                ystart = int(ROI_locs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
            elif ROI_locs[el][2] > z - int(zLength / 2):
                xstart = int(ROI_locs[el][0] - cubeLength / 2)
                ystart = int(ROI_locs[el][1] - cubeLength / 2)
                subimage = image[xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
                cubes[el].append(subimage)
    print('total time = ' + str(round(time() - start_time, 1)))
    return cubes


def key_z_plots(e):
    global curr_pos
    global background_color
    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    if zoom == 'on':
        xbegin = max([int(ROI_locs[blobNum][0]) - zoom_width, 0])
        ybegin = max([int(ROI_locs[blobNum][1]) - zoom_width, 0])
        xend = min([int(ROI_locs[blobNum][0]) + zoom_width, xpixlength])
        yend = min([int(ROI_locs[blobNum][1]) + zoom_width, ypixlength])
    elif zoom == 'off':
        xbegin = 0
        ybegin = 0
        xend = -1
        yend = -1
    curr_pos = curr_pos % len(plots)
    plt.cla()
    image = ndimage.imread(fileNames[curr_pos], flatten=True)
    image = (image - np.min(image)) / np.max(image)
    plt.imshow(image[xbegin:xend, ybegin:yend], cmap=cmaps[color_int])
    plt.gcf().gca().add_artist(r)
    plt.title('z location is: ' + str(curr_pos) + '        ' + 'z center is: ' + str(ROI_locs[blobNum][2]))
    plt.draw()
    if ROI_locs[blobNum][2] < int(zLength / 2):
        if np.abs(curr_pos) > zLength:
            background_color = 'red'
        else:
            background_color = 'gray'
    else:
        if np.abs(curr_pos - ROI_locs[blobNum][2]) > int(zLength / 2):
            background_color = 'red'
        else:
            background_color = 'gray'
    fig.patch.set_facecolor(background_color)


def key_blobs(f):
    global blobNum
    global zoom
    global xbegin
    global xend
    global ybegin
    global yend
    global curr_pos
    global background_color
    if f.key == "up":
        blobNum += 1
    elif f.key == "down":
        blobNum -= 1
    elif f.key == 'shift':
        blobNum = blobNum + 100
    elif f.key == 'control':
        blobNum = blobNum - 100
    elif f.key == '/':
        blobNum += 1
        while ROI_locs[blobNum][-1] != '?' and blobNum < len(ROI_locs) - 1:
            blobNum += 1
    else:
        return
    zoom = 'off'
    blobNum = blobNum % len(ROI_locs)
    plt.cla()
    fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
    background_color = 'gray'
    fig.patch.set_facecolor(background_color)
    curr_pos = ROI_locs[blobNum][2]
    plotInit(blobNum)


def key_zoom(h):
    global zoom
    global xbegin
    global xend
    global ybegin
    global yend
    if h.key == 'z':
        if zoom == 'off':
            zoom = 'on'
        elif zoom == 'on':
            zoom = 'off'
        plt.cla()
        fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
        fig.patch.set_facecolor(background_color)
        plotInit(blobNum)
    else:
            return


def key_tagging(g):
    global blobNum
    global curr_pos
    global fig  # added this in because there was an unresolved ref to fig (might need to do this for
    # the other key events? They are working fine now... I don't get it. )
    global color_int
    global gs
    global full_res
    global zoom
    global background_color
    if g.key == 'b' or g.key == 'n' or g.key == 'v' or g.key == 'm' or g.key == '2':
        ROI_locs[blobNum][3] = g.key
        blobNum += 1
        blobNum = blobNum % len(ROI_locs)
        zoom = 'off'
        plt.cla()
        fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
        background_color = 'gray'
        fig.patch.set_facecolor(background_color)
        curr_pos = ROI_locs[blobNum][2]
        plotInit(blobNum)
    elif g.key == '.':
        color_int += 1
        color_int = color_int % len(cmaps)
        plotInit(blobNum)
    elif g.key == ',':
        color_int += -1
        color_int = color_int % len(cmaps)
        plotInit(blobNum)
    elif g.key == 'enter':
        plt.close()
        textSaver(ROI_locs)
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
        fig.patch.set_facecolor(background_color)
        plotInit(blobNum)
        plt.show()
    elif g.key == 'escape':
        plt.close()
        textSaver(ROI_locs)
    else:
        return


def cubePlots(blobNum):
    plt.subplot(gs[0, -1])
    image = np.amax(cubes[blobNum], axis=0)
    plt.imshow(image, cmap=cmaps[color_int])
    plt.subplot(gs[1, -1])
    image = np.amax(cubes[blobNum], axis=1)
    plt.imshow(image, cmap=cmaps[color_int])
    plt.subplot(gs[2, -1])
    image = np.amax(cubes[blobNum], axis=2)
    plt.imshow(image, cmap=cmaps[color_int])

                                        ######################################################


def plotInit(blobNum):
    global gs
    global r
    global curr_pos
    global iterList
    global cmaps
    global zoom
    if zoom == 'on':
        xbegin = max([ROI_locs[blobNum][0] - zoom_width, 0])
        ybegin = max([ROI_locs[blobNum][1] - zoom_width, 0])
        xend = min([ROI_locs[blobNum][0] + zoom_width, xpixlength])
        yend = min([ROI_locs[blobNum][1] + zoom_width, ypixlength])
    elif zoom == 'off':
        xbegin = 0
        ybegin = 0
        xend = -1
        yend = -1
    cmaps = ['viridis', 'bone', 'inferno', 'BrBG', 'gist_rainbow', 'gnuplot', 'ocean', 'Paired', 'Set1']
    plt.cla()
    cubePlots(blobNum)
    plt.subplot(gs[-1, -1])
    plt.cla()
    plt.axis('off')
    plt.text(0.25, 0.5, 'label:  ' + str(ROI_locs[blobNum][-1]), fontsize=40)
    plt.subplot(gs[:, 0:4])
    fig.canvas.mpl_connect('key_press_event', key_z_plots)
    fig.canvas.mpl_connect('key_press_event', key_blobs)
    fig.canvas.mpl_connect('key_press_event', key_tagging)
    fig.canvas.mpl_connect('key_press_event', key_zoom)
    image = ndimage.imread(fileNames[curr_pos], flatten=True)
    image = (image - np.min(image)) / np.max(image)
    plt.imshow(image[xbegin:xend, ybegin:yend], cmap=cmaps[color_int])
    y, x = [ROI_locs[blobNum][i] - [xbegin, ybegin][i] for i in range(2)]
    r = patches.Rectangle((x - cubeLength/2, y - cubeLength/2), cubeLength, cubeLength, color='red', linewidth=1, fill=False)
    plt.title('z location is: ' + str(curr_pos) + '        ' + 'z center is: ' + str(ROI_locs[blobNum][2]))
    plt.gcf().gca().add_artist(r)
    plt.draw()


########################################################################################################################
#                                               SET UP
preamble()
########################################################################################################################
#                                               CREATE 3D BLOBS LIST
#                                (by looping through and blob-detecting each image)

start = 0
stop = -1
scale = 4
cubeLength = 30
zLength = 6
zoom_width = 200

blobTheBuilder(start, stop, scale)


########################################################################################################################
#                                     TRIMMING LIST OF BLOBS                                                           #

blobs = trim_segmented(blobs)
blobs = trim_consecutively(blobs)
blobs = trim_toofewtoomany(blobs)
print('Total time to trim blobs = ' + str(round(time() - trim_time, 1)))

#  blibs is one-d list of (x,y,z, bacType) for detected blobs
ROI_locs = [[blobs[i][n][0] * scale + (blobs[i][n][3] - blobs[i][n][0]) / 2 * scale,
             blobs[i][n][1] * scale + (blobs[i][n][4] - blobs[i][n][1]) / 2 * scale,
             int(i + blobs[i][n][2] / 2)] for i in range(len(blobs)) for n in range(len(blobs[i]))]
# blibs = [[blobs[i][n][0]*scale, blobs[i][n][1]*scale, int(i + blobs[i][n][2]/2)] for i in range(len(blobs))
#          for n in range(len(blobs[i]))]
ROI_locs = sorted(ROI_locs, key=lambda x: x[2])
[blip.append('?') for blip in ROI_locs]

########################################################################################################################
#                                           CUBE EXTRACTOR                                                             #
#                          ( extract a cube around each blob for classification )                                      #
#                          ( cubes is indexed by blob, z, x,y )                                                        #


cubes = cubeExtractor()
########################################################################################################################
#                                           IMAGE BUILDER                                                              #


print(str(len(cubes)) + ' detected blobs')
# for blip in blibs:
#     if blip[-1] == 'b':
#         blibs.remove(blip)
#     if blip[-1] == '?':
#         blip[-1] = 'n'

blobNum = 0
while ROI_locs[blobNum][-1] != '?' and blobNum < len(ROI_locs)-1:
    blobNum += 1
color_int = 0
xbegin = 0
ybegin = 0
xend = -1
yend = -1
curr_pos = ROI_locs[blobNum][2]
zoom = 'off'
background_color = 'gray'
full_res = 'off'
fig = plt.figure(figsize=(24, 16))
fig.suptitle('blob number ' + str(blobNum + 1), fontsize=20)
fig.patch.set_facecolor(background_color)
gs = gridspec.GridSpec(4, 5, height_ratios=[1, .5, .5, 1])
plotInit(blobNum)
plt.show()