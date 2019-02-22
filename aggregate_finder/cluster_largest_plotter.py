

from unet.data_processing import *
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, ball
from matplotlib import pyplot as plt


directory = '/media/parthasarathy/af969b3d-e298-4407-98c2-27368a8eba9f/multispecies_image_data/'
files = glob(directory + '**/*.npz', recursive=True)
files = [file for file in files if 'objs' in file]
files_di = [file for file in files if 'ae_en' in file]
files_en = [file for file in files if 'mono/en' in file]
sort_nicely(files_en)
fish = iter(files_en)
fish = [[x, next(fish)] for x in fish]
file = fish[0][1]


largest_cluster = []
for regions in fish:
    ob = []
    for file in regions:
        print(file)
        if np.load(file)['objs']!=[]:
            objs = np.load(file)['objs'][0]
            print(np.sort([np.sum(item) for item in objs]))
            ob.append(np.sort([np.sum(item) for item in objs])[-1])
        else:
            ob.append(0)

    print(ob)
    largest_cluster.append(np.max(ob))

