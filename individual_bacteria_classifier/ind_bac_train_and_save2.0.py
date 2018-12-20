

from skimage.transform import resize
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
from time import time
import tensorflow as tf
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random


def countData(train_labels, test_labels):
    totalcount = 0
    baccount = 0
    noisecount = 0
    onecount = 0
    zerocount = 0
    for i in train_labels:
        if int(i[0]) == 0:
            onecount += 1
        else:
            zerocount += 1
    print(str(onecount) + ' bacteria in training')
    print(str(zerocount) + ' noise blobs in training')
    baccount += onecount
    noisecount += zerocount
    totalcount += onecount + zerocount
    onecount = 0
    zerocount = 0
    for i in test_labels:
        if int(i[0]) == 0:
            onecount += 1
        else:
            zerocount += 1
    print(str(onecount) + ' bacteria in test')
    print(str(zerocount) + ' noise blobs in test')
    totalcount += onecount + zerocount
    baccount += onecount
    noisecount += zerocount
    print(str(totalcount) + ' objects of which ' + str(np.round(baccount/totalcount, 2)) + ' were bacteria and ' +
          str(np.round(noisecount/totalcount, 2)) + ' were noise')


def rotateData(data_in, labels, run_through=5):
    data_out = []
    for i in range(run_through):
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
            data_out.append([image, lable])
    print('done compiling rotated data')
    return data_out


def rotateData2(data_in, labels):
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


def extractDataTTSplit(filename, rot=True, testSize = 0.2):
    global num_labels
    global label_dict
    global cube_length
    labels = []
    data = []
    cubes_labels = []   # will be populated with [3d image, corresponding label]
    filenames = glob.glob(filename + '/*')
    for filename in filenames:
        print(filename)
        temp = pickle.load(open(filename, 'rb'))
        for item in temp:
            cubes_labels.append(item)
    num_labels = 0
    labels2 = []
    for line in cubes_labels:
        cube = line[0]
        adjusted_stddev = max(np.std(cube), 1.0 / np.sqrt(np.size(cube)))   # scale images by -mean/std
        cube = (cube - np.mean(cube)) / adjusted_stddev
        if int(label_dict[line[1]]) != 2:
            data.append(cube)
            labels.append(int(label_dict[line[1]]))
            if int(label_dict[line[1]]) not in labels2:
                labels2.append(int(label_dict[line[1]]))
    num_labels = len(labels2)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=testSize)
    if rot:
        rotated_data = rotateData(train_data, train_labels)
        random.shuffle(rotated_data)
        rotated_labels = [rotated_data[j][1] for j in range(len(rotated_data))]
        for el in range(len(rotated_data)):
            train_data.append(rotated_data[el][0])
            train_labels.append(rotated_labels[el])
    labels_np = np.array(train_labels).astype(dtype=np.uint8)
    train_labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)     # Put labels in one-hot
    labels_np = np.array(test_labels).astype(dtype=np.uint8)
    test_labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)      # Put labels in one-hot
    cube_length = len(train_data[0])
    return train_data, test_data, train_labels, test_labels


def extractData(filename, rot=True, shuffle=True, set_type='train'):
    global num_labels
    global label_dict
    # global cube_length
    labels = []
    data = []
    cubes_labels = []   # will be populated with [3d image, corresponding label]
    filenames = glob.glob(filename + '/*')
    for filename in filenames:
        print(filename)
        temp = pickle.load(open(filename, 'rb'))
        for item in temp:
            cubes_labels.append(item)
    if set_type == 'train':
        num_labels = 0
    labels2 = []
    if shuffle:
        random.shuffle(cubes_labels)
    for line in cubes_labels:
        cube = line[0]
        adjusted_stddev = max(np.std(cube), 1.0 / np.sqrt(np.size(cube)))   # scale images by -mean/std
        cube = (cube - np.mean(cube)) / adjusted_stddev
        if int(label_dict[line[1]]) != 2:
            data.append(cube)
            labels.append(int(label_dict[line[1]]))
            if int(label_dict[line[1]]) not in labels2:
                labels2.append(int(label_dict[line[1]]))
    if set_type == 'train':
        num_labels = len(labels2)
    if rot:
        print('rotating')
        rotated_data = rotateData(data, labels)
        random.shuffle(rotated_data)
        for el in range(len(rotated_data)):
            data.append(rotated_data[el][0])
            labels.append(rotated_data[el][1])
    # data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in data]
    # cube_length = len(data[0])*len(data[1])*len(data[2])
    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)     # Put labels in one-hot
    return data, labels

initial_time = time()
#
#                               LOAD DATA, CREATE TRAIN AND TEST SET
#
label_dict = {'b': 1, '2': 1, 'v': 1, 'n': 0, 'm': 0}
session_tf = tf.InteractiveSession()
ttsplit = True      # Set ttsplit to 'false' to train on data from \completed and test on data in \completed2
if ttsplit:
    fileloc = '/media/teddy/Bast/Teddy/TruthTables/completed'
    train_data, test_data, train_labels, test_labels = extractDataTTSplit(fileloc, rot=False)
else:
    fileloc = '/media/parthasarathy/Bast/Teddy/TruthTables/completed'
    fileloc2 = '/media/parthasarathy/Bast/Teddy/TruthTables/completed-2'
    print('Importing train data: ')
    train_data, train_labels = extractData(fileloc, rot=False, set_type='train')
    print('Importing test data: ')
    test_data, test_labels = extractData(fileloc2, rot=False, shuffle=False, set_type='test')
    print('Done importing data: ')
countData(train_labels, test_labels)
print('it took ' + str(np.round((time() - initial_time)/60, 2)) + ' minutes to get to the graph')


#                               HYPERPARAMETERS

L1 = 32  # output neurons for first layer
L2 = 64  # output neurons for second layer
L3 = 1024  # output neurons for third layer
epochs = 120  # number of times we loop through training data
batch_size = 120  # the size of the batches
l_rate = .0001  # learning rate
dropout_rate = 0.5  # rate of neurons dropped off dense layer during training
cube_length = 8 * 28 * 28
#
#                               CREATE THE TENSORFLOW GRAPH
#
pool_count = 0
flat_cube = tf.placeholder(tf.float32, shape=[None, cube_length])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
cube = tf.reshape(flat_cube, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
#   first layer
outputNeurons = cnn_3d(cube,  network_depth=3, kernel_size=[3, 3], num_kernels_init=16, keep_prob=0.5,
           final_dense_num=1024)
#   loss - optimizer - evaluation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(outputNeurons + 1e-10), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(outputNeurons, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#                               TRAIN THE NETWORK
#
train_size = len(train_data)
train_time0 = time()
session_tf.run(tf.global_variables_initializer())
print(str(epochs) + ' epochs')
ac_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    temp_data, temp_labels = rotateData2(train_data, train_labels)
    # temp_data, temp_labels = train_data, train_labels
    temp_data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in temp_data]
    for batch in range(train_size // batch_size):
        offset = (batch * batch_size) % train_size
        batch_data = temp_data[offset:(offset + batch_size)]
        batch_labels = temp_labels[offset:(offset + batch_size)]
        optimizer.run(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: dropout_rate})
        if batch % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                flat_cube: batch_data, y_: batch_labels, keep_prob: 1.0})
            print("training accuracy %g" % (train_accuracy))
            ac_list.append(train_accuracy)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.plot(ac_list)
#
#                               TEST THE TRAINED NETWORK
#



saver = tf.train.Saver()
save_path = saver.save(session_tf, "/media/teddy/Bast/Teddy/tf_models/ind_bac_model/model.ckpt")
print("Model saved in path: %s" % save_path)

