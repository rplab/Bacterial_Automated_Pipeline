

import tensorflow as tf
from time import time
from matplotlib import pyplot as plt
from unet.build_network import unet_network
import numpy as np
import gut_segmentation.local_functions as lf


#    HYPERPARAMETERS
num_classes = 2
epochs = 20
batch_size = 2
learning_rate = 0.0001
decay_rate = 1
decay_steps = 1000
momentum = 0.9
network_depth = 3
edge_loss_dict = {'3': 40, '4': 88}
downsample=4


# Determine local drive, decide whether to load or save the weights
drive = lf.drive_loc('Stephen Dedalus')
# save, save_loc = False, drive + '/Teddy/tf_models/DIC_rough_outline/model.ckpt'
# load, load_loc = False, drive + '/Teddy/tf_models/DIC_rough_outline/model.ckpt'
file_loc = drive + '/zebrafish_image_scans/bac_types/**'
train_data, test_data, train_labels, test_labels = lf.read_in_images(file_loc)
train_data = lf.pad_images(train_data, pad_to=edge_loss_dict[str(network_depth)]//2)

print(np.shape(train_data))
print(np.shape(train_labels))
# for n in range(1):
#     f, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.imshow(train_data[n])
#     ax2.imshow(train_labels[n])

# BUILD UNET
session_tf = tf.InteractiveSession()
shape_of_image = np.shape(train_data[0])
cropped_image_size = [i - edge_loss_dict[str(network_depth)] for i in shape_of_image]
input_image_0 = tf.placeholder(tf.float32, shape=[None, shape_of_image[0], shape_of_image[1]])
input_image = tf.reshape(input_image_0, [-1, shape_of_image[0], shape_of_image[1], 1])
input_mask_0 = tf.placeholder(tf.int32, shape=[None, cropped_image_size[0], cropped_image_size[1]])
input_mask = tf.one_hot(input_mask_0, depth=2, on_value=1.0, off_value=0.0, axis=-1)
unet_params = unet_network(input_image, batch_size=batch_size, network_depth=network_depth, kernel_size=[3, 3],
                           num_kernels_init=32, dropout_kept=0.8)
last_layer = unet_params["output"]
flat_prediction = tf.reshape(last_layer, [-1, 2])
flat_true = tf.reshape(input_mask, [-1, 2])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_true))
global_step = tf.Variable(0)
learning_rate_decay = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                 decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum=momentum).minimize(
    loss, global_step=global_step)

session_tf.run(tf.global_variables_initializer())
# LOAD PREVIOUS WEIGHTS
# if load:
#     session_tf = tf.InteractiveSession()
#     saver = tf.train.Saver()
#     saver.restore(session_tf, load_loc)
#     print('finished loading model')

###  TRAINING
train_size = len(train_data)
train_time0 = time()
print(str(epochs) + ' epochs')
ac_list = []
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    for batch in range(train_size // batch_size):
        offset = (batch * batch_size) % train_size
        batch_data = train_data[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]
        optimizer.run(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
        if batch % 5 == 0:
            train_loss = loss.eval(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
            prediction = last_layer.eval(feed_dict={input_image_0: batch_data})
            print("training loss %g" % (train_loss) + '  predictions: '  + str(
                  (np.unique([[[np.argmax(i) for i in j] for j in k] for k in prediction][0]))))
            ac_list.append(train_loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.figure()
plt.plot(ac_list)


test_data = lf.pad_images(test_data, pad_to=edge_loss_dict[str(network_depth)]//2)
prediction = last_layer.eval(feed_dict={input_image_0: batch_data})
predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction]

n = 1
plt.figure()
plt.imshow(batch_data[n])
plt.figure()
plt.imshow(predicted[n])
plt.figure()
plt.imshow(batch_labels[n])


# SAVE UNET
# if save:
#     saver = tf.train.Saver()
#     save_path = saver.save(session_tf, save_loc)
#     print("Model saved in path: %s" % save_path)
