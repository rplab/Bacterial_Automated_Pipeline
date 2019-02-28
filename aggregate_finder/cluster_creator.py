

from unet.data_processing import *
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, ball
from matplotlib import pyplot as plt


directory = '/media/parthasarathy/af969b3d-e298-4407-98c2-27368a8eba9f/multispecies_image_data/'
files = glob(directory + '**/*.npz', recursive=True)
files = [file for file in files if 'mask' in file]
files = [file for file in files if 'ae_en' in file]
n = 100
selem = ball(3)
sort_nicely(files)
# file = files[0]
for file in files:
    print(file)
    mask = np.load(file)
    print('1')
    mask = mask.f.arr_0
    print('2')
    objs = regionprops(label(mask))
    objs = [item for item in objs if item.area > n]
    print(len(objs))
    np.savez(file.split('mask')[0] + 'objs_region_' + file.split('region_')[1].replace('/', '_'), objs=objs)

