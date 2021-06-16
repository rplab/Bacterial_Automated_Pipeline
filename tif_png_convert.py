import imageio as io
import glob
import numpy as np
import os

image_direc = '/media/rplab/Karakoram/Multi-Species/germ free/mono/ae1/biogeog_2_2/fish1/Scans/scan_1/region_1/488nm/*.tif'

all_image_files = glob.glob(image_direc)
save_folder = all_image_files[0].split('488nm/')[0]
os.mkdir(save_folder + 'png')
save_loc = '/media/rplab/Karakoram/Multi-Species/germ free/mono/ae1/biogeog_2_2/fish1/Scans/scan_1/region_1/png/'
for image in all_image_files:
    img = io.imread(image)
    name = image.split('488nm')[1][0:-4]
    io.imsave(save_loc + name + '.png' , img)
