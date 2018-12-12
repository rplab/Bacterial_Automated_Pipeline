

import tensorflow as tf
from teddy.data_processing import *
from time import time
from matplotlib import pyplot as plt
from unet.build_network import unet_network


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def pixel_wise_softmax(input_tensor):
    exponential_map = tf.exp(input_tensor)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(input_tensor)[3]]))
    return tf.div(exponential_map, tensor_sum_exp)


def dice_loss(prediction, labels):
    print('labels: ' + str(labels.get_shape().as_list()))
    eps = 1e-5
    intersection = tf.reduce_sum(prediction * labels)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
    loss = 1 -(2 * intersection / union)
    return loss



directory_loc = '/media/parthasarathy/Bast/UNET_Projects/intestinal_outlining/DIC/rough_outline_data/Train/**'


tiled_image_size = [412, 412]
cropped_image_size = [372, 372]
train_data, test_data, train_labels, test_labels = read_in_images(directory_loc, label_string='_mask',
                                                                  size2=tiled_image_size[0],
                                                                  size1=cropped_image_size[0], test_size=0.1)
print(np.shape(train_data))
print(np.shape(train_labels))


###  HYPERPARAMETERS
num_classes = 2
epochs = 10
batch_size = 6
learning_rate = 0.01
decay_rate = 0.99
decay_steps = 100
momentum = 0.8

session_tf = tf.InteractiveSession()
input_image_0 = tf.placeholder(tf.float32, shape=[None, tiled_image_size[0], tiled_image_size[1]])
input_image = tf.reshape(input_image_0, [-1, tiled_image_size[0], tiled_image_size[1], 1])
input_mask_0 = tf.placeholder(tf.int32, shape=[None, cropped_image_size[0], cropped_image_size[1]])
input_mask = tf.one_hot(input_mask_0, depth=2, on_value=1.0, off_value=0.0, axis=-1)

# BUILD UNET
unet_params = unet_network(input_image, batch_size=batch_size, network_depth=3, kernel_size=[3, 3], num_kernels_init=16,
                           dropout_kept=0.5)
last_layer = unet_params["output"]

###  PREDICTION-LOSS-OPTIMIZER
flat_prediction = tf.reshape(last_layer, [-1, 2])
flat_true = tf.reshape(input_mask, [-1, 2])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_true))
global_step = tf.Variable(0)
learning_rate_decay = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                 decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum=momentum
                                       ).minimize(loss, global_step=global_step)
session_tf.run(tf.global_variables_initializer())

##  Restore
session_tf = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(session_tf, '/media/teddy/Bast/Teddy/tf_models/DIC_rough_outline/model.ckpt')

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
        # batch_data, batch_labels = data_augment(train_data[offset:(offset + batch_size)],
        #                                         train_labels[offset:(offset + batch_size)])
        optimizer.run(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
        if batch % 5 == 0:
            train_loss = loss.eval(feed_dict={input_image_0: batch_data, input_mask_0: batch_labels})
            print("training loss %g" % (train_loss))
            ac_list.append(train_loss)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.figure()
plt.plot(ac_list)


from skimage import io
file_loc = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/ae1/biogeog_1_1/scans/region_1/'
save_loc = '/media/parthasarathy/Stephen Dedalus/zebrafish_image_scans/test_gut_segmenter/'
files = glob(file_loc + '*.tif')
files = [file for file in files if 'mask' not in file]
sort_nicely(files)
for file in files:
    image = downscale_local_mean(ndimage.imread(file), (3, 3))
    image = (image - np.mean(image)) / np.std(image)
    size1, size2 = 372, 412
    images = tile_image(image, size1=size1, size2=size2)
    np.shape(images)
    # "testing"
    prediction = last_layer.eval(feed_dict={input_image_0: images})
    predicted = [[[np.argmax(i) for i in j] for j in k] for k in prediction]
    predicted = detile_1(image, predicted)
    io.imsave(save_loc + file.split('/')[-1],
              np.array(np.concatenate((image, abs(predicted-1)*image), axis=1), dtype='float32'), plugin='freeimage')
    # f, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(image)
    # ax2.imshow(predicted)


plt.imshow(np.concatenate((image, abs(predicted-1)*image), axis=1))

