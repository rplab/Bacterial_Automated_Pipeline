

from skimage.transform import resize
from sklearn.metrics import classification_report
from pathlib import Path
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
from time import time
import tensorflow as tf
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random


def count_data(train_labels, test_labels):
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


def drive_loc(drive_name):
    if drive_name == 'Bast':
        if str(Path.home()).split('/')[-1] == 'teddy':
            drive_name = 'Bast1'
        else:
            drive_name = 'Bast'
    drive = '/media/' + str(Path.home()).split('/')[-1] + '/' + drive_name
    return drive


def extract_data(filenames, shuffle=True, set_type='train'):
    global num_labels
    global label_dict
    labels = []
    data = []
    cubes_labels = []   # will be populated with [3d image, corresponding label]
    for filename in filenames:
        print(set_type + ':  ' + filename.split('/')[-1])
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
    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)     # Put labels in one-hot
    return data, labels


def create_val_set(data, labels, batch_size):
    temp_data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in data]
    c = list(zip(temp_data, labels))
    random.shuffle(c)
    temp_data, temp_labels = zip(*c)
    val_data = temp_data[:batch_size]
    val_labels = temp_labels[:batch_size]
    return val_data, val_labels


initial_time = time()
#                               LOAD DATA, CREATE TRAIN AND TEST SET


drive = drive_loc('Bast')
fileloc = drive + '/Teddy/TruthTables/RunThemAll'
filenames = glob.glob(fileloc + '/*')
datasets = np.unique([file.split('/')[-1][:-2] for file in filenames])
output = []
redo = True
for repeat in [0]:
    for i in datasets:
        while redo:
            redo = False
            label_dict = {'b': 1, '2': 1, 'v': 1, 'n': 0, 'm': 0}
            session_tf = tf.InteractiveSession()
            # come up with train test sets (looping through all)
            #                               LOAD DATA, CREATE TRAIN AND TEST SET
            #
            test_set_bool = [file.split('/')[-1][:-2] == i for file in filenames]
            test_set = [i for (i, v) in zip(filenames, test_set_bool) if v]
            train_sets = [i for (i, v) in zip(filenames, test_set_bool) if not v]
            print('Training Data: ')
            train_data, train_labels = extract_data(train_sets, set_type='train')
            print('Test Data: ')
            test_data, test_labels = extract_data(test_set, shuffle=False, set_type='test')

            #                               HYPERPARAMETERS

            depth = 2
            L1 = 16  # output neurons for first layer
            L_final = 1024  # output neurons for third layer
            kernel_size = [2, 5, 5]
            epochs = 120  # number of times we loop through training data
            batch_size = 120  # the size of the batches
            l_rate = .0001  # learning rate
            dropout_rate = 0.8  # rate of neurons dropped off dense layer during training
            cube_length = 8 * 28 * 28

            #                               CREATE THE TENSORFLOW GRAPH

            pool_count = 0
            flat_cube = tf.placeholder(tf.float32, shape=[None, cube_length])
            y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
            cube = tf.reshape(flat_cube, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
            keep_prob = tf.placeholder(tf.float32)
            is_train = tf.placeholder(tf.bool)
            #   first layer
            outputNeurons = cnn_3d(cube, network_depth=depth, kernel_size=kernel_size, num_kernels_init=L1,
                                   keep_prob=keep_prob, final_dense_num=L_final, is_train=is_train)
            #   loss - optimizer - evaluation
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(outputNeurons + 1e-10), reduction_indices=[1]))
            optimizer = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(outputNeurons, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #
            #                               TRAIN THE NETWORK
            val_data, val_labels = create_val_set(test_data, test_labels, batch_size)

            train_size = len(train_data)
            train_time0 = time()
            session_tf.run(tf.global_variables_initializer())
            print(str(epochs * train_size // batch_size) + ' batches')
            ac_list = []
            for epoch in range(epochs):
                print('epoch: ' + str(epoch))
                # temp_data, temp_labels = rotate_data(train_data, train_labels)
                temp_data, temp_labels = train_data, train_labels
                temp_data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in temp_data]
                for batch in range(train_size // batch_size):
                    offset = (batch * batch_size) % train_size
                    batch_data = temp_data[offset:(offset + batch_size)]
                    batch_labels = temp_labels[offset:(offset + batch_size)]
                    optimizer.run(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: dropout_rate,
                                             is_train: True})
                    if batch % 500 == 0:
                        train_accuracy = accuracy.eval(feed_dict={flat_cube: batch_data, y_: batch_labels,
                                                                   keep_prob: 1.0, is_train: False})
                        valid_accuracy = accuracy.eval(feed_dict={flat_cube: val_data, y_: val_labels,
                                                                   keep_prob: 1.0, is_train: False})
                        cross_ent = cross_entropy.eval(feed_dict={flat_cube: batch_data, y_: batch_labels,
                                                                   keep_prob: 1.0, is_train: False})
                        print('train acc: %g' % (train_accuracy) + ' val acc: %g' % (valid_accuracy) + ' cross_ent: %g' % (cross_ent))
                        ac_list.append(train_accuracy)
            if train_accuracy < 0.8:
                redo = True
            #
            #                               TEST THE TRAINED NETWORK
            #
            if redo == False:
                t0 = time()
                ac_list2 = []
                y_pred = []
                prediction = tf.argmax(outputNeurons, 1)  # translating the correct prediction from one-hot
                test_data_temp = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in test_data]
                batch_size = 1
                for batch in range(len(test_labels) // batch_size):
                    offset = (batch * batch_size) % train_size
                    batch_data = test_data_temp[offset:(offset + batch_size)]
                    batch_labels = test_labels[offset:(offset + batch_size)]
                    y_pred.append(prediction.eval(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: 1.0,
                                                             is_train: False})[0])
                t1 = time()
                print('time to classify ' + str(len(test_labels)) + ' test data = ' + str(np.round(t1 - t0, 2)) + ' seconds')
                print(str(np.round(len(test_labels) / (t1 - t0), 2)) + ' blobs per second labeled')
                true_labels = np.argmax(test_labels, 1)
                current_report = classification_report(y_pred, true_labels)
                print('accuracy = ' + str(current_report))
                output.append([ac_list, y_pred, true_labels])
            session_tf.close()
            tf.reset_default_graph()
        redo = True
    pickle.dump(output, open(drive + 'tf_single_bac_models/run_them_all/' + str(repeat), 'wb'))

