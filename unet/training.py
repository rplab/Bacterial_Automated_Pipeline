

import tensorflow as tf
import numpy as np


def pixel_wise_softmax(input_tensor):
    exponential_map = tf.exp(input_tensor)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(input_tensor)[3]]))
    return tf.div(exponential_map, tensor_sum_exp)


def dice_loss(prediction, labels):
    print('labels: ' + str(labels.get_shape().as_list()))
    eps = 1e-6
    intersection = tf.reduce_sum(prediction * labels)
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
    loss = (2 * intersection / union)
    return loss


def optimizer_func(last_layer, input_mask, learning_rate, decay_steps, decay_rate, momentum, pos_weight= 1):
    # prediction = pixel_wise_softmax(last_layer)
    # loss = dice_loss(prediction, input_mask)
    flat_prediction = tf.reshape(last_layer, [-1, 2])
    flat_true = tf.reshape(input_mask, [-1, 2])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_true))
    # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(flat_true, flat_prediction, pos_weight=pos_weight))
    global_step = tf.Variable(0)
    learning_rate_decay = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                     decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
<<<<<<< HEAD
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum=momentum).minimize(
        loss, global_step=global_step)
=======
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum=momentum
                                           ).minimize(loss, global_step=global_step)
>>>>>>> e20882350b32394bfce4217f3382f7759d005837
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum=momentum
    #                                            ).minimize(loss, global_step=global_step)
    return optimizer, loss
