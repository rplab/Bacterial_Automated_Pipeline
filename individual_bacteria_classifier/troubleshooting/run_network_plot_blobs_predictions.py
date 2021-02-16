
from skimage.transform import resize
from matplotlib import pyplot as plt
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
from time import time
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
import random
from sklearn.metrics import classification_report, confusion_matrix
tf.compat.v1.disable_eager_execution()


def shuffle(images, masks):
    """
    Zips images and masks together, randomly shuffles the order, then unzips to two lists again.
    :param images: images to be shuffled
    :param masks: masks to be shuffled
    :return: images and masks with order randomly shuffled, but still matching between the two
    """
    zipped = list(zip(images, masks))
    random.shuffle(zipped)
    images, masks = zip(*zipped)
    return images, masks

def import_data(filenames):
    """
    Imports the data and puts it in readable format, normalizes  data and converts labels to one-hot.
    :param filenames: List of filenames to import.
    :return: train_data, train_labels
    """
    global num_labels
    label_dict = {'b': 1, '2': 1, 'v': 1, 'n': 0, 'm': 0}
    labels = []
    data = []
    cubes_labels = []   # will be populated with [3d image, corresponding label]
    for filename in filenames:
        print(filename)
        temp = pickle.load(open(filename, 'rb'))
        for item in temp:
            if item:  # in one of the empty fish I think there was an empty "item"
                cubes_labels.append(item)
                if item[1] == '?':
                    print('there is unlabeled data in: ' + filename)
    labels2 = []
    print('done loading data and labels')
    std_image = []
    for line in cubes_labels:
        cube = line[0]
        # NORMALIZE DATA
        actual_std_dev = np.std(cube) / np.mean(cube)
        adjusted_stddev = max(np.std(cube), 1.0 / np.sqrt(np.size(cube)))   # scale images by -mean/std
        cube = (cube - np.mean(cube)) / adjusted_stddev
        data.append(cube)
        labels.append(int(label_dict[line[1]]))
        if int(label_dict[line[1]]) not in labels2:
            labels2.append(int(label_dict[line[1]]))
        std_image.append(actual_std_dev)
    num_labels = len(labels2)
    train_data, train_labels = shuffle(data, labels)
    labels_np = np.array(train_labels).astype(dtype=np.uint8)
    train_labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)     # Put labels in one-hot
    return train_data, train_labels, std_image

model = '/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/tensorflow_models/single_bac_models/aeromonas_mb/validation0model'
final_test = glob.glob('/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/data_and_labels/single_bac_labels/aeromonas_mb_test_data/*')

test_data, test_labels, sdev = import_data(final_test)

tf.compat.v1.reset_default_graph()
batch_size = 120  # the size of the batches
initial_kernel = 16  # number of kernels for first layer
network_depth = 2  # Number of convolutional layers
final_neurons = 1024  # number of neurons for final dense layer
kernel_size = [2, 5, 5]  # Size of kernel
cube_length = 8 * 28 * 28  # flattened size of input image

#                               CREATE THE TENSORFLOW GRAPH
flattened_image = tf.compat.v1.placeholder(tf.float32, shape=[None, cube_length])
input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
input_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, num_labels])  # I am leaving number of labels generic.
keep_prob = tf.compat.v1.placeholder(tf.float32)
#   first layer
output_neurons = cnn_3d(input_image, network_depth=network_depth, kernel_size=kernel_size,
                        num_kernels_init=initial_kernel, keep_prob=keep_prob, final_dense_num=final_neurons)
prediction = tf.argmax(input=output_neurons, axis=1)
# LOAD PREVIOUS WEIGHTS
session_tf = tf.compat.v1.InteractiveSession()
saver_bac = tf.compat.v1.train.Saver()
saver_bac.restore(session_tf, model + '/model.ckpt')

if len(test_data) > 0:
    predictions = []
    batch_size = 1

    test_data_list = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in test_data]
    mip_x = [np.amax(np.array(input_image), axis = 0) for input_image in test_data]
    mip_y = [np.amax(np.array(input_image), axis = 1) for input_image in test_data]
    mip_z = [np.amax(np.array(input_image), axis = 2) for input_image in test_data]
    max_int = [np.amax(np.array(input_image)) for input_image in test_data]

    test_prediction = tf.argmax(output_neurons, 1)
    for batch in range(len(test_labels) // batch_size):
        offset = batch
        batch_data = test_data_list[offset:(offset + batch_size)]
        batch_labels = test_labels[offset:(offset + batch_size)]
        predictions.append(
                test_prediction.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels,
                                                keep_prob: 1.0}))

    true_labels = np.argmax(test_labels, 1)

    for blob in range(len(predictions)):

        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(mip_x[blob])
        plt.subplot(1,3,2)
        plt.imshow(mip_y[blob])
        plt.subplot(1,3,3)
        plt.imshow(mip_z[blob])

        plt.suptitle('True =' + str(true_labels[blob]) + ', Prediction = ' + str(predictions[blob]) )
                     #', St Dev = ' + str(np.round(sdev[blob], 1))+ ' Max= ' + str(np.round(max_int[blob])))
        plt.savefig('/media/rplab/Aravalli/automated_pipeline/test_images_predictions/blob' + str(blob) + '.png')
        plt.close()
        if blob % 500 == 0 :
            print('Completed ' + str(blob) + 'blobs')
    print(classification_report(true_labels, np.array(predictions).flatten()))
    print(confusion_matrix(true_labels, np.array(predictions).flatten()))

