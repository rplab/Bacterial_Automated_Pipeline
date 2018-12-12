

import tensorflow as tf


def optimizer_func(last_layer, input_mask, learning_rate,
                                                     decay_steps, decay_rate, momentum):
    # prediction = pixel_wise_softmax(last_layer)
    # loss = dice_loss(prediction, input_mask)
    flat_prediction = tf.reshape(last_layer, [-1, 2])
    flat_true = tf.reshape(input_mask, [-1, 2])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_true))
    global_step = tf.Variable(0)
    learning_rate_decay = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                     decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_decay, momentum=momentum
                                           ).minimize(loss, global_step=global_step)
    return optimizer, loss