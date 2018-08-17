# Reference: https://github.com/khanrc/mnist/blob/master/inception.py 

import tensorflow as tf 

 

 

def inception_block_a(x, name='inception_a'): 

   # num of channels: 384 = 96*4 

   with tf.variable_scope(name): 

       # Pooling part 

       b1 = tf.layers.average_pooling2d(x, [3,3], 1, padding='SAME') 

       b1 = tf.layers.conv2d(inputs=b1, filters=96, kernel_size=[1, 1], padding="same", activation=tf.nn.relu) 

 

       # 1x1 part 

       b2 = tf.layers.conv2d(inputs=x, filters=96, kernel_size=[1, 1], padding="same", activation=tf.nn.relu) 

 

       # 3x3 part 

       b3 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[1, 1], padding="same", activation=tf.nn.relu) 

       b3 = tf.layers.conv2d(inputs=b3, filters=96, kernel_size=[3, 3], padding="same", activation=tf.nn.relu) 

       

       # 5x5 part 

       b4 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[1, 1], padding="same", activation=tf.nn.relu) 

       # 2 3x3 in cascade with same depth is the same as 5x5 but with less parameters 

       # b4 = tf.layers.conv2d(inputs=b4, filters=96, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) 

       b4 = tf.layers.conv2d(inputs=b4, filters=96, kernel_size=[3, 3], padding="same", activation=tf.nn.relu) 

       b4 = tf.layers.conv2d(inputs=b4, filters=96, kernel_size=[3, 3], padding="same", activation=tf.nn.relu) 

 

       concat = tf.concat([b1, b2, b3, b4], axis=-1) 

       return concat 