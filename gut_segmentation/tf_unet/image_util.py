# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from sklearn.preprocessing import OneHotEncoder
#import cv2
from skimage.util import random_noise
import glob
import numpy as np
from PIL import Image
from skimage import transform
import random
from matplotlib import pyplot as plt
from skimage.exposure import equalize_hist
from skimage.transform import downscale_local_mean

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()
        train_data = self._normalize_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self.data_aug(train_data, labels)
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]

        if self.n_class == 2:
            return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class)
        else:
            return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, labels.shape[0], labels.shape[1],
                                                                                self.n_class)

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        
        return label


    def _normalize_data(self, data):
        # normalization                                                         ####SWITCH TO SOFT NORMALIZATION####
        # data = np.clip(np.fabs(data), self.a_min, self.a_max)
        # data -= np.amin(data)
        # data /= np.amax(data)
        return data


    def data_aug(self, image, label, angle=10, resize_rate=0.9):  # need to adjust parameters to reasonable numbers
        # should add image resize here
        edge_size = 44
        npad = ((edge_size, edge_size), (edge_size, edge_size))
        npad2 = ((edge_size, edge_size), (edge_size, edge_size), (0, 0))
        # image = equalize_hist(image)
        image = downscale_local_mean(image, (2, 2))
        label = np.ceil(downscale_local_mean(label, (2, 2, 1)))
        image /= np.amax([np.amax(image), np.abs(np.amin(image))])
        image = np.pad(image, pad_width=npad, mode='reflect')
        label = np.pad(label, pad_width=npad2, mode='reflect')
        input_shape = tuple(np.ceil(np.array(image.shape) / 2.) * 2)  # make sure dimensions are even
        size = image.shape[0]
        rsize = random.randint(np.floor(resize_rate * size), size)
        w_s = random.randint(0, size - rsize)
        h_s = random.randint(0, size - rsize)
        sh = random.random() / 2 - 0.25
        rotate_angle = random.random() / 180 * np.pi * angle
        # adjust brightness, contrast, add noise.

        # cropping image
        image = image[w_s:w_s + size, h_s:h_s + size]
        label = label[w_s:w_s + size, h_s:h_s + size, :]
        # affine transform
        afine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angle)  # maybe change this to similarity transform
        image = transform.warp(image, inverse_map=afine_tf, mode='constant', cval=0)
        label = transform.warp(label, inverse_map=afine_tf, mode='constant', cval=0)
        # resize to original size
        image = transform.resize(image, input_shape, mode='constant', cval=0)
        label = transform.resize(label, input_shape, mode='constant', cval=0)
        # should add soft normalize here and simply take in non-normalized images.
        return image, label

    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
        return X, Y


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif',
                 shuffle_data=True, n_class=2, train_class=1):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.train_class = train_class
        
        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        # test = self._load_file_label(self.data_files[0].replace(self.data_suffix, self.mask_suffix), np.int)
        # print(test)
        # print([np.amin(test), np.amax(test)])
        # plt.imshow(test)
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _load_file_label(self, path, dtype=np.float32):
        if self.n_class == 2:
            return np.array(Image.open(path), np.int) >= self.train_class
        else:
            test = np.array(Image.open(path), np.int)
            test2 = np.array([(np.arange(self.n_class) == i[:, None]).astype(np.float32) for i in test])
            return test2

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_file_label(label_name, np.bool)
        return img, label

