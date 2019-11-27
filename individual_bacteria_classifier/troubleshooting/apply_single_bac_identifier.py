

from skimage.transform import resize
from individual_bacteria_classifier.build_network_3dcnn import cnn_3d
import tensorflow as tf
import numpy as np


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
input_labels = tf.placeholder(tf.float32, shape=[None, 2])  # I am leaving number of labels generic.
input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
keep_prob = tf.placeholder(tf.float32)
#   first layer
outputNeurons = cnn_3d(input_image, network_depth=depth, kernel_size=kernel_size, num_kernels_init=L1, keep_prob=keep_prob,
                       final_dense_num=L_final)
prediction = tf.argmax(outputNeurons, 1)


session_tf = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(session_tf, load_loc)



ac_list2 = []
y_pred = []
batch_size = 1
test_data = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in test_data]
for batch in range(len(test_labels) // batch_size):
    offset = batch
    print(offset)
    batch_data = test_data[offset:(offset + batch_size)]
    batch_labels = test_labels[offset:(offset + batch_size)]
    y_pred.append(prediction.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels, keep_prob: 1.0})[0])
print('time to classify ' + str(len(test_labels)) + ' test data = ' + str(np.round(t1-t0, 2)) + ' seconds')
print(str(np.round(len(test_labels)/(t1-t0), 2)) + ' blobs per second labeled')
true_labels = np.argmax(test_labels, 1)
print(classification_report(y_pred, true_labels))

# def is_it_bac_or_not(image):
#     image = [resize(np.array(image), (8, 28, 28)).flatten()]
#     y_pred = (prediction.eval(feed_dict={flat_cube: image, keep_prob: 1.0})[0])
#     return y_pred

