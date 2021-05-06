
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio as io
from accessory_functions import sort_nicely

agg_mask_direc = '/media/rplab/Aravalli/AEMB_EN_invasion_time_series/12_1_2020_ts_AEMB_dtom_EN_gfp/fish1/enterobacter/aggregates/scan_1_region_2_488nm_aggregate_mask.npz'

#agg_mask_direc = '/media/rplab/Aravalli/AEMB_EN_invasion_time_series/12_1_2020_ts_AEMB_dtom_EN_gfp/fish1/enterobacter/gutmasks/scan_1_region_1_488nm_gutmask.npz'
image_direc = glob.glob('/media/rplab/Aravalli/AEMB_EN_invasion_time_series/12_1_2020_ts_AEMB_dtom_EN_gfp/fish1/Scans/scan_1/region_2/488nm/*.tif')
sort_nicely(image_direc)
load_agg = np.load(agg_mask_direc)['gutmask']

image = [io.imread(image_direc[i]) for i in range(len(image_direc))]
save_loc = '/media/rplab/Aravalli/AEMB_EN_invasion_time_series/12_1_2020_ts_AEMB_dtom_EN_gfp/fish1/ouputs_agg/r2/488nm/'

for file in range(len(image_direc)):
    plt.figure()
    plt.imshow(image[file], cmap = 'gray')
    plt.imshow(load_agg[file], alpha = 0.4, cmap = 'gray_r')
    plt.savefig(save_loc + 'image_' + str(file))
    plt.close("all")