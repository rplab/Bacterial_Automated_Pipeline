

from matplotlib import pyplot as plt
from matplotlib import gridspec, patches
import numpy as np
import pickle
from skimage.feature import blob_dog
from skimage.measure import block_reduce
from skimage import exposure
from time import time
from scipy import ndimage
from skimage.feature import peak_local_max
import re
from scipy.ndimage.measurements import center_of_mass, label
import glob
import os.path
from skimage.filters import threshold_otsu
import imageio as io
from skimage import filters

def sort_nicely(l):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def preamble():
    global run
    global fileNames
    global fileLoc
    global bacteria_type
    global usrname
    global xpixlength
    global ypixlength
    global output_file
    global cubes
    global region
    global ROI_locs
    folder_location = input('copy paste (CTRL+SHFT+v) the file location of your first image please:  ')
    print()
    bacteria_type = input('What type of bacteria are you identifying?  ')
    fish_number = input('What fish number is this?  ')
    region = input('Which region is this?  ')
    fileLoc = folder_location
    output_loc = folder_location.split('/' + bacteria_type)[0] + '/labels/'
    output_file = output_loc + bacteria_type + '_' + fish_number + '_region_' + region
    run = 1
    if os.path.isfile(output_file):
        print()
        cubes = textLoader()[0]
        ROI_locs = textLoader()[1]
        run = 0
    fileNames = glob.glob(fileLoc + '/*.tif')
    fileNames.extend(glob.glob(fileLoc + '/*.png'))
    fileNames = [file for file in fileNames if 'gutmask' not in file]
    sort_nicely(fileNames)
    pix_dimage = io.imread(fileNames[0])[10:]
    ypixlength = len(pix_dimage[0])
    xpixlength = len(pix_dimage)


def dist(x1, y1, list):
    x2 = list[0]
    y2 = list[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def blobTheBuilder(start, stop, scale):
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

    sigma = 3

    t0 = time()
    image_3D = [io.imread(fileNames[file]) for file in range(len(fileNames))]
    t1 = time()
    t_read += t1 - t0

    mask_image = io.imread(fileLoc.split('region_1')[0] + 'Masks/region_' + str(region) +'.tif')
    gaussian_image = [ndimage.gaussian_filter(image_3D[file], sigma = 3) for file in range(len(image_3D))]

    t2 = time()
    t_reduce += t2 - t1

    t3 = time()
    thresholding_param = threshold_otsu(np.array(gaussian_image))
    print(thresholding_param)
    for files in range(len(image_3D)):
        coordinates = peak_local_max(gaussian_image[files], min_distance = 30, threshold_abs = thresholding_param,
                                     indices = False, labels = mask_image)  # outputs bool image
        ##### IF THERE ARE SEVERAL MAXIMA IN SAME REGION(multiple pixels with the same maxima adjacent to each other),
        ##### find the center of mass of these regions ####
        labels_image = label(coordinates)[0]
        labelled_centers = center_of_mass(coordinates, labels_image, range(1, np.max(labels_image) + 1))
        temp_blobs = np.array([[int(labelled_centers[n][0]), int(labelled_centers[n][1])] for n in range(len(labelled_centers))])
        blobs.append(temp_blobs)

        print('Finding blobs in image ' + str(files) + ' of ' + str(len(image_3D)))
    t4 = time()
    t_append += t4 - t3

    new_blobs = [0] * len(image_3D)
    for files in range(len(blobs)):
        list_blobs_Z = [[blobs[files][b][0], blobs[files][b][1], 1, 0, 0] for b in range(len(blobs[files]))]
        new_blobs[files] = list_blobs_Z

    blobs = new_blobs


    print('t_read = ' + str(round(t_read, 1)))
    print('t_reduce = ' + str(round(t_reduce, 1)))
    print('t_resize = ' + str(round(t_resize, 1)))
    print('t_blob = ' + str(round(t_blob, 1)))
    print('t_append = ' + str(round(t_append, 1)))


def z_trim_blobs(blobs, directory_loc):
    global image_3D
    image_3D_0 = io.imread(directory_loc[0])

    masks_blobs = [0] * len(directory_loc)

    print('determing 3D blobs')

    for i in range(len(blobs)):
        if len(blobs) > 0:

            sub_z_coordinates = [0]*len(blobs[i])

            for z in range(len(blobs[i])):
                list_coordinates = [[blobs[i][z][0] + k, blobs[i][z][1] + m] for k in range(-5, 5) for m in range(-5,5)]
                sub_z_coordinates[z] = list_coordinates

            #### flatten list of lists
            sub_z_coordinates = np.array([item for sublist in sub_z_coordinates for item in sublist])
            temp_masks = np.zeros(np.shape(image_3D_0), dtype=int)

            for b in range(len(sub_z_coordinates)):
                temp_masks[int(sub_z_coordinates[b][0]), int(sub_z_coordinates[b][1])] = 1

            masks_blobs[i] = temp_masks.astype(bool)
        else:
            masks_blobs[i] = np.empty_like(np.shape(image_3D_0), dtype = bool)


    print('labelling bacteria-like objects')
    labels_bacteria = label(masks_blobs)[0]

    image_3D = np.array([io.imread(directory_loc[file]) for file in range(len(directory_loc))])
    region_centers = center_of_mass(np.array(image_3D), labels_bacteria, range(1, np.max(labels_bacteria) + 1))

    #### obtain coordinates as x, y and z.
    coordinates = np.array(
        [[int(region_centers[n][2]), int(region_centers[n][1]), int(region_centers[n][0])] for n in range(len(region_centers))])
    image_3D = []
    return coordinates

def upgraded_cube_extractor():
    z_length = 10
    image_3D = np.array([io.imread(image) for image in directory_loc])
    cubes = []
    for z_center in ROI_locs:
        if ((z_center[2] > z_length / 2) and (z_center[2] < len(image_3D) - z_length / 2)):

            xstart = int(z_center[0] - cubeLength / 2)
            ystart = int(z_center[1] - cubeLength / 2)
            zstart = int(z_center[2] - z_length / 2)
            subimage = image_3D[zstart : zstart + z_length, xstart:xstart + cubeLength, ystart:ystart + cubeLength].tolist()
            cubes.append(subimage)
    return cubes

def textSaver(blibs):
    global cubes2
    cubes2 = [[] for element in cubes]
    print('saving...')
    for el in range(len(blibs)):
        cubes2[el] = [cubes[el], blibs[el][3], blibs[el][0:3]]
    pickle.dump(cubes2, open(output_file, 'wb'))

    print('done saving truth table')

def textLoader():
    loaded = pickle.load(open(output_file, 'rb'))
    cubes1 = []
    blibs1 = []
    for el in loaded:
        cubes1.append(el[0])
        blibs1.append([el[2:5][0][0], el[2:5][0][1], el[2:5][0][2], el[1]])
    return [cubes1, blibs1]

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
    curr_pos = curr_pos % len(fileNames)
    plt.cla()
    image = io.imread(fileNames[curr_pos])[10:]
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
        xbegin = int(max([ROI_locs[blobNum][0] - zoom_width, 0]))
        ybegin = int(max([ROI_locs[blobNum][1] - zoom_width, 0]))
        xend = int(min([ROI_locs[blobNum][0] + zoom_width, xpixlength]))
        yend = int(min([ROI_locs[blobNum][1] + zoom_width, ypixlength]))
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
    image =io.imread(fileNames[curr_pos])[10:]
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
scale = 1
cubeLength = 30
zLength = 10
zoom_width = 200
if run == 1:

    blobTheBuilder(start, stop, scale)


    ########################################################################################################################
    #                                     TRIMMING LIST OF BLOBS                                                           #
    directory_loc = glob.glob(fileLoc + '/*.tif')

    sort_nicely(directory_loc)

    final_centers = z_trim_blobs(blobs, directory_loc)

    ROI_locs = [list(i) for i in final_centers]
    ROI_locs = [[ROI_locs[i][1], ROI_locs[i][0], ROI_locs[i][2]]for i in range(len(ROI_locs)) if
                (ROI_locs[i][2] > zLength / 2 and ROI_locs[i][2] < len(directory_loc) - zLength / 2)
                ]
    ROI_locs = list(sorted(ROI_locs, key=lambda x: x[2]))
    [blip.append('?') for blip in ROI_locs]

    ########################################################################################################################
    #                                           CUBE EXTRACTOR                                                             #
    #                          ( extract a input_image around each blob for classification )                                      #
    #                          ( cubes is indexed by blob, z, x,y )                                                        #
    #cubes = cubeExtractor()
    cubes = upgraded_cube_extractor()

else:
    plots = []
    for name in fileNames[start:stop]:
        image = io.imread(name)[10:]
        image = block_reduce(image, block_size=(scale, scale), func=np.mean)
        print(name)
        plots.append(image.tolist())
########################################################################################################################
#                                           IMAGE BUILDER                                                              #


print(str(len(cubes)) + ' detected blobs')

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

#######################################

### segmenting bacteria finding z center

# for cube in range(500): #range(len(cubes)):
#     plt.figure()
#     plt.subplot(311)
#     plt.imshow(np.amax(cubes[cube], axis=0))
#     plt.subplot(312)
#     plt.imshow(np.amax(cubes[cube], axis=1))
#     plt.subplot(313)
#     plt.imshow(np.amax(cubes[cube], axis=2))
#     plt.savefig('/media/rplab/Aravalli/local_max_fixed_blobs/blob_' + str(cube))
#     plt.close()
#
