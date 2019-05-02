


import numpy as np
from skimage import transform
import random



###  DATA PROCESSING  --
# *** CHECK AND MODIFY AS NEEDED ***
def data_augment(images, masks, angle=5, resize_rate=0.9):  # Adjust params. Combine with data read-in in training?
    for i in range(len(masks)):
        image = images[i]/np.amax([np.amax(images[i]), np.abs(np.amin(images[i]))])
        mask = masks[i]
        # should add image resize here

        input_shape_image = image.shape
        input_shape_mask = mask.shape
        size = image.shape[0]
        rsize = random.randint(np.floor(resize_rate * size), size)
        w_s = random.randint(0, size - rsize)
        h_s = random.randint(0, size - rsize)
        sh = random.random() / 2 - 0.25
        rotate_angle = random.random() / 180 * np.pi * angle
        # adjust brightness, contrast, add noise.

        # cropping image
        image = image[w_s:w_s + size, h_s:h_s + size]
        mask = mask[w_s:w_s + size, h_s:h_s + size]
        # affine transform
        afine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angle)  # maybe change this to similarity transform
        image = transform.warp(image, inverse_map=afine_tf, mode='constant', cval=0)
        mask = transform.warp(mask, inverse_map=afine_tf, mode='constant', cval=0)
        # resize to original size
        image = transform.resize(image, input_shape_image, mode='constant', cval=0)
        mask = transform.resize(mask, input_shape_mask, mode='constant', cval=0)
        # should add soft normalize here and simply take in non-normalized images.
        images[i], masks[i] = image, mask
    images = [(sub_image - np.mean(sub_image)) / np.std(sub_image) for sub_image in images]
    return images, masks
