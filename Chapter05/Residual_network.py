# Reference 

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/resnet.py 

import tensorflow as tf 

from collections import namedtuple 

 

# Configurations for each bottleneck group. 

BottleneckGroup = namedtuple('BottleneckGroup', 

                            ['num_blocks', 'num_filters', 'bottleneck_size']) 

groups = [ 

   BottleneckGroup(3, 128, 32), BottleneckGroup(3, 256, 64), 

   BottleneckGroup(3, 512, 128), BottleneckGroup(3, 1024, 256) 

] 

 

# Create the bottleneck groups, each of which contains `num_blocks` 

# bottleneck groups. 

for group_i, group in enumerate(groups): 

   for block_i in range(group.num_blocks): 

       name = 'group_%d/block_%d' % (group_i, block_i) 

 

       # 1x1 convolution responsible for reducing dimension 

       with tf.variable_scope(name + '/conv_in'): 

           conv = tf.layers.conv2d( 

               net, 

               filters=group.num_filters, 

               kernel_size=1, 

               padding='valid', 

               activation=tf.nn.relu) 

           conv = tf.layers.batch_normalization(conv, training=training) 

 

       with tf.variable_scope(name + '/conv_bottleneck'): 

           conv = tf.layers.conv2d( 

               conv, 

               filters=group.bottleneck_size, 

               kernel_size=3, 

               padding='same', 

               activation=tf.nn.relu) 

           conv = tf.layers.batch_normalization(conv, training=training) 

 

       # 1x1 convolution responsible for restoring dimension 

       with tf.variable_scope(name + '/conv_out'): 

           input_dim = net.get_shape()[-1].value 

           conv = tf.layers.conv2d( 

               conv, 

               filters=input_dim, 

               kernel_size=1, 

               padding='valid', 

               activation=tf.nn.relu) 

           conv = tf.layers.batch_normalization(conv, training=training) 

 

       # shortcut connections that turn the network into its counterpart 

       # residual function (identity shortcut) 

       net = conv + net 