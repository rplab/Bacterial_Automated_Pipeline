

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
        label = labels[el]
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
        lab.append(label)
    return data, lab


def import_data(filenames, testSize=0):
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
        # NORMALIZE DATA
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


initial_time = time()

#   LOAD DATA, CREATE TRAIN AND TEST SET

# Set location of images, location to save model, and the bacteria to train on
file_loc = '/home/chiron/Documents/single_bac_labels'
save, save_loc = True, '/home/chiron/Documents/single_bac_models'
train_on = 'enterobacter'


#   HYPERPARAMETERS
depth = 2  # Number of convolutional layers
L1 = 16  # number of kernels for first layer
L_final = 1024  # number of neurons for final dense layer
kernel_size = [2, 5, 5]  # Size of kernel
epochs = 120  # number of times we loop through training data
batch_size = 120  # the size of the batches
l_rate = .00001  # learning rate
dropout_rate = 0.5  # rate of neurons dropped off dense layer during training
cube_length = 8 * 28 * 28  # flattened size of input image


# Start with all bacteria, remove the one we are interested in
bacteria_set = {'aeromonas01', 'enterobacter', 'plesiomonas', 'pseudomonas', 'vibrio_z20', 'cholera', 'empty'}
bacteria_set.remove(train_on)
files = glob.glob(file_loc + '/**/*')  # Get all files
files = [file for file in files if any([bac in file for bac in bacteria_set])]  # Keep files from the correct bacteria
train_data, test_data, train_labels, test_labels = import_data(files, testSize=0)

# Print how many bacteria and how many not-bacteria are in training data
print('In initial training set:')
count_data(train_labels)


#   CREATE THE TENSORFLOW GRAPH

pool_count = 0
session_tf = tf.InteractiveSession()
flattened_image = tf.placeholder(tf.float32, shape=[None, cube_length])
input_labels = tf.placeholder(tf.float32, shape=[None, num_labels])  # I am leaving number of labels generic.
input_image = tf.reshape(flattened_image, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
keep_prob = tf.placeholder(tf.float32)
#   first layer
output_neurons = cnn_3d(input_image, network_depth=depth, kernel_size=kernel_size, num_kernels_init=L1, keep_prob=keep_prob,
                        final_dense_num=L_final)
#   loss - optimizer - evaluation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(input_labels * tf.log(output_neurons + 1e-10), reduction_indices=[1]))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BATCH NORM
with tf.control_dependencies(update_ops):  # BATCH NORM
    train_op = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)  # BATCH NORM
correct_prediction = tf.equal(tf.argmax(output_neurons, 1), tf.argmax(input_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#   TRAIN THE NETWORK

train_size = len(train_data)
train_time0 = time()
session_tf.run(tf.global_variables_initializer())   # CHECK TO SEE IF REMOVE FOR TRANSFER LEARNING
print(str(epochs) + ' epochs')
loss_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    temp_data, temp_labels = rotate_data(train_data, train_labels)
    temp_data = [resize(np.array(image), (8, 28, 28)).flatten() for image in temp_data]  # SHOULD CROP INSTEAD
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
            prediction = tf.argmax(output_neurons, 1).eval(feed_dict={flattened_image: batch_data,
                                                                      input_labels: batch_labels, keep_prob: 1.0})
            print("cross entropy = %g" % loss + "|| accuracy = " + str(training_accuracy) +
                  '  ||  predicting ' + str(np.unique(prediction)))
            loss_list.append(loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to complete initial training on network')
N = 5
plt.plot(np.convolve(loss_list, np.ones((N,)) / N, mode='valid'))
plt.xlabel('time')
plt.xlabel('cross entropy')


#   RETRAIN THE NETWORK

# find all files, and keep only the ones with the desired bacteria
files = glob.glob(file_loc + '/**/*')
files = [file for file in files if train_on in file]
train_data, test_data, train_labels, test_labels = import_data(files, testSize=0)

# Print how many bacteria and how many not-bacteria are in training data
print('In transfer training set:')
count_data(train_labels)


train_size = len(train_data)
train_time1 = time()
# session_tf.run(tf.global_variables_initializer())   # CHECK TO SEE IF REMOVE FOR TRANSFER LEARNING
epochs = epochs//2
print(str(epochs) + ' epochs')
loss_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    temp_data, temp_labels = rotate_data(train_data, train_labels)
    temp_data = [resize(np.array(image), (8, 28, 28)).flatten() for image in temp_data]  # SHOULD CROP INSTEAD
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
            prediction = tf.argmax(output_neurons, 1).eval(feed_dict={flattened_image: batch_data,
                                                                      input_labels: batch_labels, keep_prob: 1.0})
            print("cross entropy = %g" % loss + "|| accuracy = " + str(training_accuracy) +
                  '  ||  predicting ' + str(np.unique(prediction)))
            loss_list.append(loss)
print('it took ' + str(np.round((time() - train_time1) / 60, 2)) + ' minutes to train final network')
print('and ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes in total')
N = 5
plt.figure()
plt.plot(np.convolve(loss_list, np.ones((N,)) / N, mode='valid'))
plt.xlabel('time')
plt.xlabel('cross entropy')

if save:
    saver = tf.train.Saver()
    save_path = saver.save(session_tf, save_loc + '/' + train_on + '/model/model.ckpt')
    print("Model saved in path: %s" % save_path)


if len(test_data) > 0:
    ac_list2 = []
    y_pred = []
    batch_size = 1
    test_data = [resize(np.array(input_image), (8, 28, 28)).flatten() for input_image in test_data]
    test_prediction = tf.argmax(output_neurons, 1)
    for batch in range(len(test_labels) // batch_size):
        offset = batch
        batch_data = test_data[offset:(offset + batch_size)]
        batch_labels = test_labels[offset:(offset + batch_size)]
        y_pred.append(test_prediction.eval(feed_dict={flattened_image: batch_data, input_labels: batch_labels,
                                                      keep_prob: 1.0}))
    true_labels = np.argmax(test_labels, 1)
    print(classification_report(np.array(y_pred).flatten(), true_labels))
