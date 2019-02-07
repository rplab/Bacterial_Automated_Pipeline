


from unet.data_processing import *
from matplotlib import pyplot as plt
from glob import glob
from skimage.feature import blob_dog

drive = drive_loc('Stephen Dedalus')

directory = drive + '/Multi-Species/Multi-Species/germ free/five/' \
           '11_7_2018 - (EN-gfp AE-rfp)/B/Fish2/fish1/Scans/scan_1/region_1/568nm'
files = glob(directory + '/*')

files = [file for file in files if 'mask' not in file]
sort_nicely(files)
images = []
for file in files:
    image = plt.imread(file)
    images.append(image)
print('done loading images')

plt.imshow(np.amax(images, axis=0))
plt.imshow(images[27])

output = blob_dog(images[27], min_sigma=10, max_sigma=100, threshold=0.02)

