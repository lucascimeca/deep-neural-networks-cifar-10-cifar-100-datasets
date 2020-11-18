#!/disk/scratch/mlp/miniconda2/bin/python

import os
import datetime
import time
import sys
import itertools
import numpy as np
import tensorflow as tf
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider

# check necessary environment variables are defined
assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')

# load data
train_data = CIFAR100DataProvider('train', batch_size=50, shuffle_order=False)
valid_data = CIFAR100DataProvider('valid', batch_size=50, shuffle_order=False)
train_data_coarse = CIFAR100DataProvider('train', batch_size=50, use_coarse_targets=True, shuffle_order=False)
train_data.inputs = train_data.inputs.reshape(train_data.inputs.shape[0], 32, 32, 3)
valid_data.inputs = valid_data.inputs.reshape(valid_data.inputs.shape[0], 32, 32, 3)

valid_inputs = valid_data.inputs
valid_targets = valid_data.to_one_of_k(valid_data.targets)

# ---------------- define helper functions -------------------------------------------------------------------
def fully_connected_layer(l_inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim) ** 0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(l_inputs, weights) + biases)
    return outputs


def conv_layer(l_inputs, input_channel_dim, output_channel_dim, kernel_size=5, bias_init=0.0, name=""):
    kernel = tf.Variable(
        tf.truncated_normal(
            shape=[kernel_size,
                   kernel_size,
                   input_channel_dim,
                   output_channel_dim],
            stddev=5e-2,
            name='weights'))

    conv = tf.nn.conv2d(
        l_inputs,
        kernel,
        [1, 1, 1, 1],
        padding='SAME'
    )
    biases = tf.Variable(tf.zeros([output_channel_dim]), 'biases')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name="conv" + name)
    return conv1


def pool_layer(l_inputs, kernel_size=[1, 3, 3, 1], strides_dim=[1, 2, 2, 1], name=""):
    return tf.nn.max_pool(l_inputs, ksize=kernel_size,
                          strides=strides_dim,
                          padding='SAME', name='pool' + name)


def norm_layer(l_inputs, depth_radius=4, bias=1, alpha=0.001 / 9.0, beta=0.75, name=""):
    return tf.nn.lrn(l_inputs, depth_radius=depth_radius,
                     bias=bias, alpha=alpha, beta=beta,
                     name='norm' + name)


def reshape_layer(l_inputs, output_dim, batch_size, bias_init=0.1, name=""):
    reshape = tf.reshape(l_inputs, tf.pack([batch_size, -1]))
    dim = l_inputs.get_shape()[1].value*l_inputs.get_shape()[2].value*l_inputs.get_shape()[3].value
    weights = tf.Variable(tf.truncated_normal(
                                [dim, output_dim],
                                stddev=5e-2),
                          name='weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    return tf.nn.relu(tf.matmul(reshape, weights) + biases, name="reshape"+name)

#-----------------------------------------------------------------------------------------------------
#---------------------- define model graph------------------------------------------------------------

output_dim=train_data.num_classes

tf.reset_default_graph()
batch_size = tf.placeholder(tf.int32)
inputs = tf.placeholder(tf.float32, [None,
                                     train_data.inputs.shape[1],
                                     train_data.inputs.shape[2],
                                     train_data.inputs.shape[3]
                                    ], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
targets_coarse = tf.placeholder(tf.float32, [None, train_data_coarse.num_classes], 'target_coarse')

# ----------------------------- NETWORK DEFINITION --------------------------------------------------
with tf.name_scope('conv-layer-1'):
    conv1 = conv_layer(inputs,
                       input_channel_dim=train_data.inputs.shape[3],
                       output_channel_dim=80,
                       kernel_size=5,
                       name="1")
with tf.name_scope('pool-layer-1'):
    pool1 = pool_layer(conv1, name="1")
with tf.name_scope('norm-layer-1'):
    norm1 = norm_layer(pool1)
with tf.name_scope('conv-layer-2'):
    conv2 = conv_layer(norm1,
                       input_channel_dim=norm1.get_shape().as_list()[3],
                       output_channel_dim=80,
                       kernel_size=5,
                       bias_init=0.1,
                       name="2")
with tf.name_scope('pool-layer-2'):
    pool2 = pool_layer(conv2, name="2")
with tf.name_scope('fully_connected_layer1'):
    fully_connected_layer1 = reshape_layer(pool2, 400, batch_size)
with tf.name_scope('fully_connected_layer2'):
    fully_connected_layer2 = fully_connected_layer(fully_connected_layer1, 400, 200)
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fully_connected_layer2, 200, train_data.num_classes, nonlinearity=tf.identity)
with tf.name_scope('output-layer-coarse'):
    outputs_coarse = fully_connected_layer(fully_connected_layer2, 200, train_data_coarse.num_classes, nonlinearity=tf.identity)
# ------------ define error computation -------------
with tf.name_scope('error'):
    vars = tf.trainable_variables()
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets) +
                           tf.add_n([ tf.nn.l2_loss(v) for v in vars
                                      if 'bias' not in v.name ]) * 0.005)
with tf.name_scope('error-coarse'):
    error_coarse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs_coarse, targets_coarse) +
                           tf.add_n([ tf.nn.l2_loss(v) for v in vars
                                      if 'bias' not in v.name ]) * 0.005)
with tf.name_scope('error-multitask'):
    error_multitask = tf.reduce_mean(error + error_coarse)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
with tf.name_scope('accuracy-coarse'):
    accuracy_coarse = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs_coarse, 1), tf.argmax(targets_coarse, 1)),
            tf.float32))
with tf.name_scope('accuracy-multitask'):
    accuracy_multitask = tf.reduce_mean(accuracy + accuracy_coarse)
# --- define training rule ---
with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(error)
with tf.name_scope('train-multitask'):
    train_step_multitask = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(error_multitask)


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

# add summary operations
tf.summary.scalar('error', error)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

# create objects for writing summaries and checkpoints during training
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(os.environ['OUTPUT_DIR'], timestamp)
checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))
saver = tf.train.Saver()

# create arrays to store run train / valid set stats
num_epoch = 1000
train_accuracy = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)
valid_accuracy = np.zeros(num_epoch)
valid_error = np.zeros(num_epoch)

# create session and run training loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())
step = 0
max_valid_accuracy = (None, 0)
trainining = train_step_multitask
for e in range(num_epoch):
    current_time = time.time()
    for (input_batch, target_batch), (_, target_batch_coarse) \
            in itertools.izip_longest(train_data, train_data_coarse):
        # do train step with current batch
        _, summary, batch_error, batch_acc = sess.run(
            [training, summary_op, error, accuracy],
            feed_dict={inputs: input_batch,
                       targets: target_batch,
                       targets_coarse: target_batch_coarse,
                       batch_size: target_batch.shape[0]}
        )
        # add symmary and accumulate stats
        train_writer.add_summary(summary, step)
        train_error[e] += batch_error
        train_accuracy[e] += batch_acc
        step += 1
    # normalise running means by number of batches
    train_error[e] /= train_data.num_batches
    train_accuracy[e] /= train_data.num_batches
    # evaluate validation set performance
    valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
        [summary_op, error, accuracy],
        feed_dict={inputs: valid_inputs,
                   targets: valid_targets,
                   batch_size: valid_targets.shape[0]})
    valid_writer.add_summary(valid_summary, step)
    # checkpoint model variables
    saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), step)
    # write stats summary to stdout
    print('-------- time elapsed: {}sec -------'.format(time.time()-current_time))
    print('Epoch {0:02d}: err(train)={1:.6f} acc(train)={2:.6f}'
          .format(e + 1, train_error[e], train_accuracy[e]))
    print('          err(valid)={0:.6f} acc(valid)={1:.6f}'
          .format(valid_error[e], valid_accuracy[e]))
    sys.stdout.flush()
    # early stopping
    if max_valid_accuracy[1] < valid_accuracy[e]:
        max_valid_accuracy = (e, valid_accuracy[e])
    elif e > max_valid_accuracy[0] + 10 and training=train_step_multitask:
        training = train_step
        print('CHANGING TRAINING STEP!!')
    else: 
       break

# close writer and session objects
train_writer.close()
valid_writer.close()
sess.close()

# save run stats to a .npz file
np.savez_compressed(
    os.path.join(exp_dir, 'run.npz'),
    train_error=train_error,
    train_accuracy=train_accuracy,
    valid_error=valid_error,
    valid_accuracy=valid_accuracy
)
