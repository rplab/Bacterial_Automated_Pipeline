


from tf_unet import unet
from skimage import io
from skimage.util import pad
from skimage.transform import downscale_local_mean, resize
from scipy.misc import imread
from glob import glob
from time import time
import matplotlib.pyplot as plt
import numpy as np


def segment_gut(image, edge_size=44):
    image /= np.amax(image)
    image = np.pad(image, [edge_size, edge_size], mode='reflect')
    data = np.array([[[[i] for i in j] for j in image]])
    prediction = net.predict(path, data)
    predicted_mask = [[np.argmax(i) for i in j] for j in prediction[0]]
    x_pad, y_pad = np.int_(np.subtract(np.shape(image), np.shape(predicted_mask)) / 2)
    predicted_mask_padded = pad(predicted_mask, ((x_pad, x_pad), (y_pad, y_pad)), 'constant', constant_values=0)
    return image * predicted_mask_padded


net = unet.Unet(layers=4, features_root=32, channels=1, n_class=2, cost_kwargs=dict(summaries=False))

computer = '/media/teddy/'
path = computer + 'Bast/Teddy/tf_models/gut_outline_model/model.cpkt'


# test_loc = computer + 'Bast/Teddy/UNET_Projects/intestinal_outlining/Fluorescence/finished_testing_data/*.tif'
# files = glob(test_loc)
# test_files = [i for i in files if 'mask' not in i]
# test_masks = [i for i in files if 'mask' in i]
# print(test_files)
# print(test_masks)
# test_num = 0
# edge_size = 44
# for test_num in range(len(test_files)):
#     image = imread(test_files[test_num])
#     image = downscale_local_mean(image, (2, 2))
#     input_shape = tuple(np.ceil(np.array(image.shape) / 2.) * 2)
#     image /= np.amax(image)
#     image = resize(image, input_shape, mode='constant', cval=0)
#     image /= np.amax(image)
#     image = np.pad(image, [edge_size, edge_size],
#                              mode='reflect')
#     data = np.array([[[[i] for i in j] for j in image]])
#     time0 = time()
#     prediction = net.predict(path, data)
#     print(time() - time0)
#
#     predicted_mask = [[np.argmax(i) for i in j] for j in prediction[0]]
#     actual_image = [[i+1 for i in j] for j in io.imread(test_files[test_num])]
#     actual_mask = io.imread(test_masks[test_num])
#     x_pad, y_pad = np.int_(np.subtract(np.shape(actual_image), np.shape(predicted_mask))/2)
#     predicted_mask_padded = pad(predicted_mask, ((x_pad, x_pad), (y_pad, y_pad)), 'constant', constant_values=0)
#
#     fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
#     ax1.set_title('actual image')
#     ax1.imshow(actual_image)
#     ax1.axis('off')
#     ax2.set_title('actual_mask * actual image')
#     ax2.imshow(actual_mask * actual_image)
#     ax2.axis('off')
#     ax3.set_title('computer\'s mask of the actual image')
#     ax3.imshow(predicted_mask_padded * actual_image)
#     ax3.axis('off')

