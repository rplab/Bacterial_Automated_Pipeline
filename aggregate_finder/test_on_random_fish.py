


from scipy.misc import imsave
from unet.data_processing import *
import random
from matplotlib import pyplot as plt
from glob import glob
import os
import re


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


drive = drive_loc('Stephen Dedalus')
directory = drive + '/zebrafish_image_scans/bac_types/'
bac_type_directories = glob(directory + '*/')

region_dict = {'1': 1200, '2': 800}

images = []
names = []
for folder in bac_type_directories:
    name = folder.split('/')[-2]
    random_fish = random.choice(glob(folder + '*/'))
    name += '--' + random_fish.split('/')[-2]
    # random_region = random.choice(glob(random_fish + 'scans/*/'))
    random_region = random_fish + 'scans/region1'
    name += '--' + random_region.split('/')[-2]
    files = glob(random_region + '/*.tif')
    files = [file for file in files if 'mask' not in file]
    sort_nicely(files)
    names.append(name)
    images.append([])
    for file in files:
        image = plt.imread(file)
        images[-1].append(image[10:])
    print('done loading images')

save_directory = drive + '/zebrafish_image_scans/test_agg_finder/'
save_directories = [save_directory + i.split('/')[-2] for i in bac_type_directories]
filelist = glob(os.path.join(save_directory + '/**/', "*.tif"))
for f in filelist:
    os.remove(f)
incr = 0
for image_set in images:
    # mip = np.amax(image_set, axis=0)
    # thresh = np.mean(mip) + 3 * np.std(mip)
    thresh = 1200
    print(save_directories[incr])
    incr2 = 0
    for image in image_set:
        imsave(save_directories[incr] + '/' + str(incr2) + '.tif',
               np.concatenate((image, (image > thresh) * image), axis=1))
        if incr2 % 10 == 0:
            print(str(int(np.round(incr2/len(image_set), 2)*100)) + '% done')
        incr2 += 1
    incr += 1

