
from glob import glob
from accessory_functions import mask_blob_trim
import pickle
import os
from time import time
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
import numpy as np
from skimage.transform import resize


#
# The ordering is experiment / fish / scans / regions / color
# Input         colors, bacs, directory
# Output        color1:  (xyz, t)
#               color2:  (xyz, t)
#

def est_files(directory):
    '''Takes in the usrs directory location and finds all images.  Outputs the associated xyzt textfile,
    trained model and the filenames (and addreess) of the images.'''
    if not os.path.exists(str(directory) + 'automation_data'):
        os.mkdir(str(directory) + 'automation_data')
    #  # SWITCH TO ONLY CREATE IF DOESN'T ALREADY EXIST
    f = [open(str(directory) + 'automation_data/xyzt0.dat', 'ab'),
         open(str(directory) + 'automation_data/xyzt1.dat', 'ab')]
    filenames = glob(str(directory) + '**/*.tif', recursive=True)
    filenames.extend(glob(str(directory) + '**/*.png', recursive=True))
    filenames = sorted(filenames, key=lambda x: float(x.split('nm')[0][-3]))   #  Sort data by color
    filenames = [[item for item in filenames if '488' in item], [item for item in filenames if '568' in item]]
    clf = pickle.load(open('RandomForestModel.pkl', 'rb'))
                # clf depends on bacterial type of color which needs to be an input
    return f, clf, filenames


def resize_normalize_image(image):
    if np.shape(image) == (10, 30, 30):
        image = np.array(image)[1:-1, 1:-1, 1:-1]
    else:
        image = resize(np.array(cubes[k]), (8, 28, 28), ).flatten()
    image = (image - np.mean(image))/np.std(image)
    return image




file_loc = '/media/teddy/Bast1/Teddy/single_bac_labeled_data/single_bac_labels/'
load_loc = '/media/teddy/Bast1/Teddy/single_bac_labeled_data/tf_single_bac_models'

#                               HYPERPARAMETERS

depth = 2  # Number of convolutional layers
L1 = 16  # number of kernels for first layer
L_final = 1024  # number of neurons for final dense layer
kernel_size = [2, 5, 5]  # Size of kernel
batch_size = 120  # the size of the batches
l_rate = .0001  # learning rate
dropout_rate = 0.5  # rate of neurons dropped off dense layer during training
cube_length = 8 * 28 * 28  # flattened size of input image

#                               CREATE THE TENSORFLOW GRAPH

session_tf = tf.InteractiveSession()
flattened_image = tf.placeholder(tf.float32, shape=[None, cube_length])
input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
keep_prob = tf.placeholder(tf.float32)
#   first layer
outputNeurons = cnn_3d(input_image, network_depth=depth, kernel_size=kernel_size, num_kernels_init=L1,
                       keep_prob=keep_prob, final_dense_num=L_final)
prediction = tf.argmax(outputNeurons, 1)

saver = tf.train.Saver()
saver.restore(session_tf, load_loc)


directory = 'D:/Teddy/AV_competition_7_16_15/'
# directory = '/media/teddyhay/Bast/Teddy/AV_competition_7_16_15/'
f, clf, filenames = est_files(directory)

beginning = time()
numbugs = np.zeros(len(filenames))
for color_num in range(len(filenames)):
    files = filenames[color_num]
    i = 0
    j = 0
    while i < len(files) - 1:   # loops through all folders in the dataset
                                # Output is:
        #   Creates a sorted list of the next fish / scan / region / image Num  Then calls j=i
        tempList = []
        while files[i].split('pco')[0] == files[j].split('pco')[0] and i < len(files) - 1:
            tempList.append(files[i])
            i += 1
        j = i
        tempList = sorted(tempList, key=lambda x: float(x.split('pco')[1].split('.')[0]))
        # Identify fish, time, region
        fish_num = tempList[0].split('fish')[1][0]
        region_num = tempList[0].split('region_')[1][0]
        time_num = tempList[0].split('scan_')[1][0]
        #  Call the mask building blob detecting machine learning function here.
        #  Output is: cubes [extracted 30X30X10 cubes], blibs [x,y,z, bacType]
        cubes, ROI_locs = mask_blob_trim(tempList)

        for k in range(len(ROI_locs)):
            image = resize_normalize_image(image)
            predicted = prediction.eval(feed_dict={flattened_image: image, keep_prob: 1.0})[0]
            ROI_locs[k].append(predicted)
        # Save all of the [cubes, blibs] independently as well as add to the xyzt text file for each color
        pickle.dump([cubes, ROI_locs], open(str(directory) + 'automation_data/c:' + str(color_num) + 'f:' +
                                            str(fish_num) + 't:' + str(time_num) + 'r:' + str(region_num), 'wb'))
        xyzt = [item[:2] + [time_num] for item in ROI_locs if item[3] == 1]
                                # Need to adjust positions in xyzt for each region. Finish Stitch Together.
        np.savetxt(f[color_num], xyzt)  # need to change this so that it saves for each color separately.
        numbugs[color_num] += len([blips for blips in ROI_locs if blips[3] == 1])  # Make this a function of fish, scan, bac

print(str(numbugs) + ' total bacteria')
print('total time = ' + str(round(time() - beginning, 1)))
