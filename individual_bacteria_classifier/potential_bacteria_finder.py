
import numpy as np
from skimage.feature import blob_dog
from skimage.measure import block_reduce
from skimage import exposure
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Squelch all info messages.

cube_length = 28
z_length = 8


def dist(x1, y1, compare_list):
    x2 = compare_list[0]
    y2 = compare_list[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def difference_of_gaussians_2D(images, scale, min_sig=2, max_sig=20, thresh=0.02):
    plots = []
    blobs = []
    for image in images:
        image = block_reduce(image, block_size=(scale, scale), func=np.mean)  # This could just be downscale local mean?
        plots.append(image.tolist())
        image = (image - np.min(image))/np.max(image)
        temp_blobs = blob_dog(image, max_sigma=max_sig, min_sigma=min_sig, threshold=thresh, overlap=0).tolist()
        for temp_blob in temp_blobs:
            temp_blob.append(0)
            temp_blob.append(0)
        if not temp_blobs:
            blobs.append([[]])
        else:
            blobs.append(temp_blobs)
    return blobs, plots


def segmentation_mask(plots1, width, thresh2):
    plots_out = [[] for el in range(len(plots1))]
    plots2 = [[] for el in range(len(plots1))]
    for i in range(len(plots1)):
        image = plots1[i]
        image = (image - np.min(image))/np.max(image)
        plots2[i] = exposure.equalize_hist(np.array(image))
    for i in range(len(plots2)):
        if i < int(width / 2):
            image = np.mean(plots2[0: width], axis=0)
        elif i > int(len(plots2) - width / 2):
            image = np.mean(plots2[-width: -1], axis=0)
        else:
            image = np.mean(plots2[i-int(width / 2):i + int(width / 2)], axis=0)
        binary = image > thresh2
        plots_out[i] = binary
    return plots_out


def trim_segmented(blobs, plots, width=30, thresh2=0.7):
    plots1 = segmentation_mask(plots, width, thresh2)
    for z in range(len(blobs)):
        rem = []
        for blob in blobs[z]:
            if plots1[z][int(blob[0])][int(blob[1])] is False and blob != []:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


# Loop through blobs trimming consecutive blobs
def trim_consecutively(blobs, adjSize=2):
    for z in range(len(blobs)):
        for n in range(len(blobs[z])):
            if blobs[z][n][2] == 0:
                break
            else:
                blobs[z][n][2] = 1
                contains = 'True'
                zz = z + 1
                test_location = blobs[z][n][0:2]
                while contains == 'True' and zz < len(blobs):
                    if not blobs[zz]:  # check for empty zz
                        break
                    for blob in blobs[zz]:
                        if dist(blob[0], blob[1], test_location) < adjSize:
                            blobs[z][n][2] += 1
                            test_location = blob[0:2]
                            # x-end
                            blobs[z][n][3] = test_location[0]
                            # x-end
                            blobs[z][n][4] = test_location[1]

                            blobs[zz].remove(blob)
                            zz += 1
                            contains = 'True'
                            break
                        else:
                            contains = 'False'
    return blobs


#  trim when blob only in one or two planes
def trim_toofewtoomany(blobs, ypixlength, xpixlength, tooFew=2, tooMany=15):
    for z in range(len(blobs)):
        rem = []    # note, removing while looping skips every other entry to be removed
        for blob in blobs[z]:
            if blob[2] < tooFew or blob[2] > tooMany:
                rem.append(blob)
            # the following makes sure blobs aren't on x-y edge of image
            elif blob[0] < cube_length or blob[1] < cube_length:
                rem.append(blob)
            elif blob[0] > xpixlength - cube_length:
                rem.append(blob)
            elif blob[1] > ypixlength - cube_length:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


def cube_extractor(extracted_ROI, images, blobs):
    z = 0
    cubes = [[] for i in extracted_ROI]
    for image in images:
        z += 1
        for el in range(len(extracted_ROI)):
            if extracted_ROI[el][2] > len(blobs) - int(z_length / 2) and z > len(blobs) - z_length:
                x_start = int(extracted_ROI[el][0] - cube_length / 2)
                y_start = int(extracted_ROI[el][1] - cube_length / 2)
                sub_image = image[x_start:x_start + cube_length, y_start:y_start + cube_length].tolist()
                cubes[el].append(sub_image)
            elif extracted_ROI[el][2] > z + int(z_length / 2):
                break
            elif extracted_ROI[el][2] <= int(z_length / 2) and z <= z_length:
                x_start = int(extracted_ROI[el][0] - cube_length / 2)
                y_start = int(extracted_ROI[el][1] - cube_length / 2)
                sub_image = image[x_start:x_start + cube_length, y_start:y_start + cube_length].tolist()
                cubes[el].append(sub_image)
            elif extracted_ROI[el][2] > z - int(z_length / 2):
                x_start = int(extracted_ROI[el][0] - cube_length / 2)
                y_start = int(extracted_ROI[el][1] - cube_length / 2)
                sub_image = image[x_start:x_start + cube_length, y_start:y_start + cube_length].tolist()
                cubes[el].append(sub_image)
    return cubes


def blob_the_builder(images):
    """
    Function to roughly find blobs that might be bacteria and extract a cube of pixels around each one
    :param images: A 3D stack of images of a gut
    :return: potential_bacteria_voxels - a list of 30x30x10 voxels containing potential bacteria
             blob_locs - a list of locations of each of the potential bacteria voxels
    """

    ypixlength = np.shape(images)[1]
    xpixlength = np.shape(images)[2]
    scale = 4  # should this be hard coded? If it is an input we can scale it for each bacteria?
    blobs, plots = difference_of_gaussians_2D(images, scale)  # Also, more inputs that are being left as defaults

    # Trim the very rough selection of blobs
    blobs = trim_segmented(blobs, plots)  # remove detected objects outside of crude approximation of the gut
    blobs = trim_consecutively(blobs)  # stitch together detected objects along the z-dimension
    blobs = trim_toofewtoomany(blobs, ypixlength, xpixlength)  # remove blobs that are two short or long in z

    # blob_locs is a one-d list of (x,y,z) for detected blobs
    blob_locs = [[blobs[i][n][0] * scale + (blobs[i][n][3] - blobs[i][n][0]) / 2 * scale,
                 blobs[i][n][1] * scale + (blobs[i][n][4] - blobs[i][n][1]) / 2 * scale,
                 int(i + blobs[i][n][2] / 2)] for i in range(len(blobs)) for n in range(len(blobs[i]))]
    blob_locs = sorted(blob_locs, key=lambda x: x[2])

    # Extract a cube around each blob for classification
    potential_bacteria_voxels = cube_extractor(blob_locs, images, blobs)
    potential_bacteria_voxels = [(image - np.mean(image)) / np.std(image) for image in potential_bacteria_voxels]
    return potential_bacteria_voxels, blob_locs
