

import tensorflow as tf
import unet.data_processing as dp
import unet.training as train
from time import time
from matplotlib import pyplot as plt
from unet.build_network import unet_network
import numpy as np


###  HYPERPARAMETERS
num_classes = 2
epochs = 4
batch_size = 2
learning_rate = 0.01
decay_rate = 1
decay_steps = 100
momentum = 0.9
network_depth = 3
tiled_image_size = [412, 412]
edge_loss_dict = {'3': 40, '4': 88}
cropped_image_size = [i - edge_loss_dict[str(network_depth)] for i in tiled_image_size]


# Determine local drive, decide whether to load or save the weights
drive = dp.drive_loc('Stephen Dedalus')
save, save_loc = False, drive + '/Teddy/tf_models/fluor-gut/model.ckpt'
load, load_loc = False, drive + '/Teddy/tf_models/fluor-gut/model.ckpt'
# LOAD IN DATA
directory_loc = drive + '/zebrafish_image_scans/bac_types/**'
train_data, test_data, train_labels, test_labels = dp.read_in_images(directory_loc, label_string='_gutmask',
                                                                  size2=tiled_image_size[0],
                                                                  size1=cropped_image_size[0], test_size=0.1,
                                                                  import_length=10000, downsample=2)


# BUILD UNET
session_tf = tf.InteractiveSession()
input_image_0 = tf.placeholder(tf.float32, shape=[None, tiled_image_size[0], tiled_image_size[1]])
input_image = tf.reshape(input_image_0, [-1, tiled_image_size[0], tiled_image_size[1], 1])
input_mask_0 = tf.placeholder(tf.int32, shape=[None, cropped_image_size[0], cropped_image_size[1]])
input_mask = tf.one_hot(input_mask_0, depth=2, on_value=1.0, off_value=0.0, axis=-1)
unet_params = unet_network(input_image, batch_size=batch_size, network_depth=network_depth, kernel_size=[3, 3],
                           num_kernels_init=16, dropout_kept=0.8)
last_layer = unet_params["output"]

###  PREDICTION-LOSS-OPTIMIZER
optimizer, loss = train.optimizer_func(last_layer, input_mask, momentum=momentum, learning_rate=learning_rate,
                                       decay_steps=decay_steps, decay_rate=decay_rate)
session_tf.run(tf.global_variables_initializer())

# LOAD PREVIOUS WEIGHTS
if load:
    session_tf = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(session_tf, load_loc)
    print('finished loading model')

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
        batch_data, batch_labels = dp.data_augment(train_data[offset:(offset + batch_size)],
                                                   train_labels[offset:(offset + batch_size)])
        optimizer.run(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
        if batch % 5 == 0:
            train_loss = loss.eval(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
            print("training loss %g" % (train_loss))
            ac_list.append(train_loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.figure()
plt.plot(ac_list)

# SAVE UNET
if save:
    saver = tf.train.Saver()
    save_path = saver.save(session_tf, save_loc)
    print("Model saved in path: %s" % save_path)



from skimage import io
save_loc = drive + '/UNET_Projects/intestinal_outlining/Fluorescence/prediction/'
for save_int in range(len(test_data) // batch_size):
    offset = (save_int * batch_size) % train_size
    image = test_data[offset:(offset + batch_size)]
    prediction = last_layer.eval(feed_dict={input_image_0: image})
    predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction]
    predicted = np.concatenate((predicted[0], predicted[1]), axis=0)
    np.shape(predicted)
    np.shape(image)
    image = np.concatenate((image[0], image[1]), axis=0)
    io.imsave(save_loc + str(save_int),
              np.array(np.concatenate((image, abs(predicted)*image), axis=1), dtype='float32'), plugin='freeimage')
print('done saving')

plt.imshow(np.concatenate((image, abs(predicted-1)*image), axis=1))

