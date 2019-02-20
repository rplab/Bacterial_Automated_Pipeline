

from unet.data_processing import *
from skimage.measure import label, regionprops


directory = '/media/parthasarathy/af969b3d-e298-4407-98c2-27368a8eba9f/multispecies_image_data/'
files = glob(directory + '**/*.npz', recursive=True)
files = [file for file in files if 'mask' in file]
n = 100
sort_nicely(files)
for file in files:
    mask = np.load(file)
    objs_raw = regionprops(label(mask))
    objs = [item for item in objs_raw if item.area > n]


