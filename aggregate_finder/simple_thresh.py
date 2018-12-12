



from scipy.misc import imsave
from unet.data_processing import *
from matplotlib import pyplot as plt
from time import time


drive = drive_loc('Stephen Dedalus')
directory = drive + '/zebrafish_image_scans/bac_types/ae1/biogeog_1_3/scans/region_2/'
save_directory = drive + '/bac cluster_segment_test/ae1_f1_3_r2/'

time_init = time()
files = glob(directory + '/*.tif')
files = [file for file in files if 'mask' not in file]
sort_nicely(files)
images = []
for file in files:
    image = plt.imread(file)
    images.append(image)
print('done loading images')

mip = np.amax(images, axis=0)
thresh = np.mean(mip) + 3 * np.std(mip)



for i in range(len(images)):
    imsave(save_directory + str(i) + '.tif', np.concatenate((images[i], (images[i] > thresh) * images[i]), axis=1))
    if i%10 == 0:
        print(str(int(np.round(i/len(images), 2)*100)) + '% done')
print('done')

