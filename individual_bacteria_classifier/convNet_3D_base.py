

from skimage.transform import resize
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
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


def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3D(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # added 1 more stride for 3d


def convLayer(x, kernel=[2, 5, 5], numIn = 1, numOut = 16):
    W_conv1 = weightVariable([kernel[0], kernel[1], kernel[2], numIn, numOut])
    b_conv1 = biasVariable([numOut])
    return tf.nn.leaky_relu(conv3D(x, W_conv1) + b_conv1)


def maxPool2x2(x):
    global pool_count
    pool_count += 1
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME') # what is difference btwn ksize and stride?


def denseLayer(x, numIn=2 * 7 * 7 * 32, numOut = 1024):
    W_fc1 = weightVariable([numIn, numOut])
    b_fc1 = biasVariable([numOut])
    h_pool2_flat = tf.reshape(x, [-1, numIn])
    dense =  tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return dense


def softmaxLayer(x, numIn=1024, numLabels=2):
    W_fc2 = weightVariable([numIn, numLabels])
    b_fc2 = biasVariable([numLabels])
    return tf.nn.softmax(tf.matmul(x, W_fc2) + b_fc2)


def rotate_data(data_in, labels):
    data = []
    lab = []
    for el in range(len(data_in)):
        image = data_in[el]
        lable = labels[el]
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


def read_in_data(file_loc, bacterium='virbio_z20'):
    global num_labels
    global label_dict
    global cube_length
    labels = []
    data = []
    cubes_labels = []   # will be populated with [3d image, corresponding label]
    files = glob.glob(file_loc + '/*/*')
    files = [file for file in files if bacterium not in file]
    for file in files:
        print(file)
        temp = pickle.load(open(file, 'rb'))
        for item in temp:
            if item:  # This is to make sure that no empty lists are included.
                cubes_labels.append(item)
                #  The following code checks for ?s in case someone did not fully label a dataset
                # if item[1] == '?':
                #     print('????  ' + file)
    num_labels = 0
    labels2 = []
    for line in cubes_labels:
        cube = line[0]
        adjusted_stddev = max(np.std(cube), 1.0 / np.sqrt(np.size(cube)))   # scale images by -mean/std
        cube = (cube - np.mean(cube)) / adjusted_stddev
        data.append(cube)
        labels.append(int(label_dict[line[1]]))
        if int(label_dict[line[1]]) not in labels2:
            labels2.append(int(label_dict[line[1]]))
    num_labels = len(labels2)
    print('data and labels created')

    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)     # Put labels in one-hot
    cube_length = len(data[0])
    return data, labels


initial_time = time()
#
#                               LOAD DATA, CREATE TRAIN AND TEST SET
#
label_dict = {'b': 1, '2': 1, 'v': 1, 'n': 0, 'm': 0}
session_tf = tf.InteractiveSession()
file_loc = '/media/teddy/Bast1/Teddy/single_bac_labeled_data/bacteria_types'
bacteria = ['aeromonas01', 'cholera', 'vibrio_z20', 'pseudomonas', 'plesiomonas', 'enterobacter']
data, labels = read_in_data(file_loc, bacterium=bacteria[0])


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
conv_l1 = convLayer(cube, numIn=1, numOut=L1)  # numIn inp
# ut size, numOut is output size.
#   pooling
pooling_l1 = maxPool2x2(conv_l1)
#   second layer
conv_l2 = convLayer(pooling_l1, numIn=L1, numOut=L2)
#   pooling
pooling_l2 = maxPool2x2(conv_l2)
#   dense layer - neurons fully connecting all conv neuron outputs
dense_neurons = L3
dense_l3 = denseLayer(pooling_l2, numIn=int(cube_length / (2 * pool_count) ** 3) * L2,
                      numOut=dense_neurons)
keep_prob = tf.placeholder(tf.float32)
dropped_l3 = tf.nn.dropout(dense_l3, keep_prob)
#   softmax
outputNeurons = softmaxLayer(dropped_l3, numIn=dense_neurons, numLabels=num_labels)  # soft max to predict
#   loss - optimizer - evaluation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(outputNeurons + 1e-10), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(outputNeurons, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#                               TRAIN THE NETWORK
#
train_size = len(data)
train_time0 = time()
session_tf.run(tf.global_variables_initializer())
print(str(epochs) + ' epochs')
ac_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    temp_data, temp_labels = rotate_data(data, labels)
    temp_data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in temp_data]
    for batch in range(train_size // batch_size):
        offset = (batch * batch_size) % train_size
        batch_data = temp_data[offset:(offset + batch_size)]
        batch_labels = temp_labels[offset:(offset + batch_size)]
        optimizer.run(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: dropout_rate})
        if batch % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: 1.0})
            print("training accuracy %g" % (train_accuracy))
            ac_list.append(train_accuracy)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.plot(ac_list)


#
#                               SAVE THE TRAINED NETWORK
#
saver = tf.train.Saver()
save_path = saver.save(session_tf,
                       "/media/teddy/Bast1/Teddy/single_bac_labeled_data/models_single_bac/aeromonas01/model.ckpt")
print("Model saved in path: %s" % save_path)

