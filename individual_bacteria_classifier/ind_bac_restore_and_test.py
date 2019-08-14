






from skimage.transform import resize
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
import numpy as np
from pathlib import Path


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


def drive_loc(drive_name):
    if drive_name == 'Bast':
        if str(Path.home()).split('/')[-1] == 'teddy':
            drive_name = 'Bast1'
        else:
            drive_name = 'Bast'
    drive = '/media/' + str(Path.home()).split('/')[-1] + '/' + drive_name
    return drive

drive = drive_loc('Bast')
save_loc = drive + '/Teddy/tf_single_bac_models/z20/model.ckpt'


#                               HYPERPARAMETERS

depth = 2
L1 = 8  # output neurons for first layer
L_final = 1024  # output neurons for third layer
kernel_size = [2, 3, 3]
epochs = 10  # number of times we loop through training data
batch_size = 20  # the size of the batches
l_rate = .0001  # learning rate
dropout_rate = 0.7  # rate of neurons dropped off dense layer during training
cube_length = 8 * 28 * 28

#                               CREATE THE TENSORFLOW GRAPH

pool_count = 0
flat_cube = tf.placeholder(tf.float32, shape=[None, cube_length])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
cube = tf.reshape(flat_cube, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
keep_prob = tf.placeholder(tf.float32)
#   first layer
outputNeurons = cnn_3d(cube,  network_depth=depth, kernel_size=kernel_size, num_kernels_init=L1, keep_prob=keep_prob,
           final_dense_num=L_final)
prediction = tf.argmax(outputNeurons, 1)


session_tf = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(session_tf, save_loc)



# ac_list2 = []
# y_pred = []
# batch_size = 1
# test_data = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in test_data]
# t0 = time()
# for batch in range(len(test_labels) // batch_size):
#     offset = batch
#     print(offset)
#     batch_data = test_data[offset:(offset + batch_size)]
#     batch_labels = test_labels[offset:(offset + batch_size)]
#     y_pred.append(prediction.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels, keep_prob: 1.0})[0])
# t1 = time()
# print('time to classify ' + str(len(test_labels)) + ' test data = ' + str(np.round(t1-t0, 2)) + ' seconds')
# print(str(np.round(len(test_labels)/(t1-t0), 2)) + ' blobs per second labeled')
# true_labels = np.argmax(test_labels, 1)
# print(classification_report(y_pred, true_labels))

def is_it_bac_or_not(image):
    image = [resize(np.array(image), (8, 28, 28)).flatten()]
    y_pred = (prediction.eval(feed_dict={flat_cube: image, keep_prob: 1.0})[0])
    return y_pred

