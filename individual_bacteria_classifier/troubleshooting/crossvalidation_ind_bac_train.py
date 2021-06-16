

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



def count_data(train_labels):
    onecount = 0
    zerocount = 0
    for i in train_labels:
        if int(i[0]) == 0:
            onecount += 1
        else:
            zerocount += 1
    print(str(onecount) + ' bacteria in training')
    print(str(zerocount) + ' noise blobs in training')




def rotate_data(data_in, labels):
    data = []
    lab = []
    for el in range(len(data_in)):
        image = data_in[el]
        lable = labels[el]
        # if len(image[0]) == 30 and len(image[1]) == 30:
        if np.random.randint(0, 2) == 0:
            image = np.fliplr(image)
        if np.random.randint(0, 2) == 0:
            image = np.flipud(image)
        if np.random.randint(0, 2) == 0:
            image = np.array(image)[:, :, ::-1]
        if np.random.randint(0, 2) == 0:
            image = np.transpose(image, (0, 2, 1))
        data.append(image)
        lab.append(lable)
    return data, lab


def import_data_2(filenames, testSize=0):
    """
    Imports the data and puts it in readable format, normalizes  data and converts labels to one-hot.
    :param filenames: List of filenames to import.
    :param testSize: For sklearn's train test split. Likely zero.
    :return: train_data, test_data, train_labels, test_labels
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
    for line in cubes_labels:
        cube = line[0]
        #### NORMALIZE DATA
        adjusted_stddev = max(np.std(cube), 1.0 / np.sqrt(np.size(cube)))   # scale images by -mean/std
        cube = (cube - np.mean(cube)) / adjusted_stddev
        data.append(cube)
        labels.append(int(label_dict[line[1]]))
        if int(label_dict[line[1]]) not in labels2:
            labels2.append(int(label_dict[line[1]]))
    num_labels = len(labels2)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=testSize)
    labels_np = np.array(train_labels).astype(dtype=np.uint8)
    train_labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)     # Put labels in one-hot
    labels_np = np.array(test_labels).astype(dtype=np.uint8)
    test_labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)      # Put labels in one-hot
    return train_data, test_data, train_labels, test_labels


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
    for line in cubes_labels:
        cube = line[0]
        # NORMALIZE DATA
        adjusted_stddev = max(np.std(cube), 1.0 / np.sqrt(np.size(cube)))   # scale images by -mean/std
        cube = (cube - np.mean(cube)) / adjusted_stddev
        data.append(cube)
        labels.append(int(label_dict[line[1]]))
        if int(label_dict[line[1]]) not in labels2:
            labels2.append(int(label_dict[line[1]]))
    num_labels = len(labels2)
    train_data, train_labels = shuffle(data, labels)
    labels_np = np.array(train_labels).astype(dtype=np.uint8)
    train_labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)     # Put labels in one-hot
    return train_data, train_labels



initial_time = time()
#
#                               LOAD DATA, CREATE TRAIN AND TEST SET
#
file_loc = '/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/data_and_labels/single_bac_labels/'
save, save_loc = True, '/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/tensorflow_models/single_bac_models/aeromonas_mb/'
load, load_loc = True, '/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/tensorflow_models/single_bac_models/vibrio_z20/'
bacteria_dict = {'aeromonas_mb', 'aemb_cross_val', 'vib_ae', 'enterobacter', 'plesiomonas', 'pseudomonas', 'vibrio_z20', 'cholera', 'empty'}
included_bacteria = ['aemb_cross_val']  # List of all bacteria to be included in training data
files = glob.glob(file_loc + '/**/*')
files = [file for file in files if any([bac in file for bac in included_bacteria])]  # Only use filenames with our
# bacteria

#                               HYPERPARAMETERS

depth = 2  # Number of convolutional layers
L1 = 16  # number of kernels for first layer
L_final = 1024  # number of neurons for final dense layer
kernel_size = [2, 5, 5]  # Size of kernel
epochs = 60  # number of times we loop through training data
batch_size = 120  # the size of the batches
l_rate = .00001  # learning rate
dropout_rate = 0.5  # rate of neurons dropped off dense layer during training
cube_length = 8 * 28 * 28  # flattened size of input image
num_labels = 2
#                               CREATE THE TENSORFLOW GRAPH

pool_count = 0
session_tf = tf.compat.v1.InteractiveSession()
flattened_image = tf.compat.v1.placeholder(tf.float32, shape=[None, cube_length])
input_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, num_labels])  # I am leaving number of labels generic.
input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
keep_prob = tf.compat.v1.placeholder(tf.float32)
#   first layer
output_neurons = cnn_3d(input_image, network_depth=depth, kernel_size=kernel_size, num_kernels_init=L1, keep_prob=keep_prob,
                        final_dense_num=L_final)
#   loss - optimizer - evaluation
cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=input_labels * tf.math.log(output_neurons + 1e-10), axis=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=input_labels, logits = tf.math.log(output_neurons + 1e-10), pos_weight=1))

update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)  #  BATCH NORM
with tf.control_dependencies(update_ops):  #  BATCH NORM
    train_op = tf.compat.v1.train.AdamOptimizer(l_rate).minimize(cross_entropy)  #  BATCH NORM
correct_prediction = tf.equal(tf.argmax(input=output_neurons, axis=1), tf.argmax(input=input_labels, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

if load:
    saver = tf.compat.v1.train.Saver()
    saver.restore(session_tf, load_loc + 'model/model.ckpt')
#
#
#                               TRAIN THE NETWORK

train_time0 = time()
session_tf.run(tf.compat.v1.global_variables_initializer())

for file in range(1,len(files)):
    validation_file = files[file]
    training_files = [i for i in files if i not in validation_file]

    train_data, train_labels = import_data(training_files)
    count_data(train_labels)
    test_data, test_labels = import_data([validation_file])

    batch_size = 120
    train_size = len(train_data)
  # CHECK TO SEE IF REMOVE FOR TRANSFER LEARNING
    print(str(epochs) + ' epochs')
    loss_list = []

    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        temp_data, temp_labels = rotate_data(train_data, train_labels)
        temp_data = [resize(np.array(image), (8, 28, 28)).flatten() for image in temp_data]  ### SHOULD CROP INSTEAD
        for batch in range(train_size // batch_size):

            offset = (batch * batch_size) % train_size
            batch_data = temp_data[offset:(offset + batch_size)]
            batch_labels = temp_labels[offset:(offset + batch_size)]
            train_op.run(feed_dict={flattened_image: batch_data, input_labels: batch_labels, keep_prob: dropout_rate})
            if batch % 50 == 0:
                loss = cross_entropy.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels,
                                                     keep_prob: 1.0})
                training_accuracy = accuracy.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels,
                                                             keep_prob: 1.0})
                prediction = tf.argmax(input=output_neurons, axis=1).eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels,
                                                                          keep_prob: 1.0})
                print("cross entropy = %g" % loss + "|| accuracy = " + str(training_accuracy) + '  ||  predicting ' + str(np.unique(prediction)))
                loss_list.append(loss)

    if len(test_data) > 0:
        predictions = []
        batch_size = 1
        test_data = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in test_data]
        test_prediction = tf.argmax(output_neurons, 1)
        for batch in range(len(test_labels) // batch_size):
            offset = batch
            batch_data = test_data[offset:(offset + batch_size)]
            batch_labels = test_labels[offset:(offset + batch_size)]
            predictions.append(
                test_prediction.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels,
                                                keep_prob: 1.0}))
        true_labels = np.argmax(test_labels, 1)

        print(classification_report(np.array(predictions).flatten(), true_labels))
        print(confusion_matrix(np.array(predictions).flatten(), true_labels))



if save:
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(session_tf, save_loc + 'model/model.ckpt')
    print("Model saved in path: %s" % save_path)


    print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
    N = 5
    plt.plot(np.convolve(loss_list, np.ones((N,)) / N, mode='valid'))
    plt.xlabel('time')
    plt.xlabel('cross entropy')

final_test = glob.glob('/media/rplab/Stephen Dedalus/automated_pipeline_labels_models/data_and_labels/single_bac_labels/aemb_test_data/*')

test_data, test_labels = import_data(final_test)

# tf.compat.v1.reset_default_graph()
# batch_size = 120  # the size of the batches
# initial_kernel = 16  # number of kernels for first layer
# network_depth = 2  # Number of convolutional layers
# final_neurons = 1024  # number of neurons for final dense layer
# kernel_size = [2, 5, 5]  # Size of kernel
# cube_length = 8 * 28 * 28  # flattened size of input image
#
# #                               CREATE THE TENSORFLOW GRAPH
# flattened_image = tf.compat.v1.placeholder(tf.float32, shape=[None, cube_length])
# input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
# input_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, num_labels])  # I am leaving number of labels generic.
# keep_prob = tf.compat.v1.placeholder(tf.float32)
# #   first layer
# output_neurons = cnn_3d(input_image, network_depth=network_depth, kernel_size=kernel_size,
#                         num_kernels_init=initial_kernel, keep_prob=keep_prob, final_dense_num=final_neurons)
# prediction = tf.argmax(input=output_neurons, axis=1)
# # LOAD PREVIOUS WEIGHTS
# session_tf = tf.compat.v1.InteractiveSession()
# saver_bac = tf.compat.v1.train.Saver()
# saver_bac.restore(session_tf, load_loc + '/model/model.ckpt')



if len(test_data) > 0:
    predictions = []
    batch_size = 1
    test_data = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in test_data]
    test_prediction = tf.argmax(output_neurons, 1)
    for batch in range(len(test_labels) // batch_size):
        offset = batch
        batch_data = test_data[offset:(offset + batch_size)]
        batch_labels = test_labels[offset:(offset + batch_size)]
        predictions.append(
            test_prediction.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels,
                                            keep_prob: 1.0}))
    true_labels = np.argmax(test_labels, 1)

    print(classification_report(np.array(predictions).flatten(), true_labels))
    print(confusion_matrix(np.array(predictions).flatten(), true_labels))