






from skimage.transform import resize
from sklearn.metrics import classification_report
from time import time
import tensorflow as tf
import numpy as np


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


#                               HYPERPARAMETERS

L1 = 32  # output neurons for first layer
L2 = 64  # output neurons for second layer
L3 = 1024  # output neurons for third layer
epochs = 1  # number of times we loop through training data
batch_size = 120  # the size of the batches
l_rate = .0001  # learning rate
dropout_rate = 0.5  # rate of neurons dropped off dense layer during training
cube_length = 8 * 28 * 28
num_labels = 2

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
prediction = tf.argmax(outputNeurons, 1)


session_tf = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(session_tf, '/media/teddy/Bast/Teddy/tf_models/ind_bac_model/model.ckpt')



# ac_list2 = []
# y_pred = []
# batch_size = 1
# test_data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in test_data]
# t0 = time()
# for batch in range(len(test_labels) // batch_size):
#     offset = batch
#     print(offset)
#     batch_data = test_data[offset:(offset + batch_size)]
#     batch_labels = test_labels[offset:(offset + batch_size)]
#     y_pred.append(prediction.eval(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: 1.0})[0])
# t1 = time()
# print('time to classify ' + str(len(test_labels)) + ' test data = ' + str(np.round(t1-t0, 2)) + ' seconds')
# print(str(np.round(len(test_labels)/(t1-t0), 2)) + ' blobs per second labeled')
# true_labels = np.argmax(test_labels, 1)
# print(classification_report(y_pred, true_labels))

def is_it_bac_or_not(image):
    image = [resize(np.array(image), (8, 28, 28)).flatten()]
    y_pred = (prediction.eval(feed_dict={flat_cube: image, keep_prob: 1.0})[0])
    return y_pred

