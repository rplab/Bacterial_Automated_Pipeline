

from matplotlib import pyplot as plt
from accessory_functions import sort_nicely
from glob import glob
import numpy as np


directory = '/media/parthasarathy/af969b3d-e298-4407-98c2-27368a8eba9f/multispecies_image_data/'
files = glob(directory + '**/*.npz', recursive=True)
files = [file for file in files if 'mask_' in file]
files_di = [file for file in files if 'ae_en' in file]
files_di_en = [file for file in files_di if '488' in file]
sort_nicely(files_di_en)
files_en = [file for file in files if 'mono/en' in file]
sort_nicely(files_en)
fish_di_en = iter(files_di_en)
fish_di_en = [[x, next(fish_di_en)] for x in fish_di_en]
files_di_ae = [file for file in files_di if '568' in file]
sort_nicely(files_di_ae)
files_ae = [file for file in files if 'mono/a01' in file]
sort_nicely(files_ae)
fish_di_ae = iter(files_di_ae)
fish_di_ae = [[x, next(fish_di_ae)] for x in fish_di_ae]

temp_file = files_en[1]
image_files = glob('/'.join(temp_file.split('/')[:-1]) + '/scans/**/*.tif')
image_files = [file for file in image_files if temp_file.split('mask_')[-1][:8] in file]
sort_nicely(image_files)
mask_load = np.load(temp_file)
mask = mask_load.f.arr_0
mask_max_proj = np.amax(mask, axis=0)
images = []
for file in image_files:
    images.append(plt.imread(file))
image = np.amax(images, axis=0)
plt.imshow(mask_max_proj)
plt.imshow(image)
