# Reference:
# https://github.com/exelban/tensorflow-cifar-10
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
# https://stackoverflow.com/questions/34471563/logging-training-and-validation-loss-in-tensorboard
import fire
import numpy as np
import os
import tensorflow as tf


class Train:
    __x_ = []
    __y_ = []
    __logits = []
    __loss = []
    __train_step = []
    __merged_summary_op = []
    __saver = []
    __session = []
    __writer = []
    __is_training = []
    __loss_val = []
    __train_summary = []
    __val_summary = []

    def __init__(self):
        pass

    def build_graph(self):
        self.__x_ = tf.placeholder("float", shape=[None, 32, 32, 3], name='X')
        self.__y_ = tf.placeholder("int32", shape=[None, 10], name='Y')
        # Add dropout to the fully connected layer
        self.__is_training = tf.placeholder(tf.bool)
        with tf.name_scope("model") as scope:
            conv1 = tf.layers.conv2d(inputs=self.__x_, filters=64, kernel_size=[5, 5],
                                     padding="same", activation=None) # tf.nn.relu

            conv1_bn = tf.layers.batch_normalization(inputs=conv1, axis=-1, momentum=0.9, epsilon=0.001, center=True,
                                                     scale=True, training=self.__is_training, name='conv1_bn')
            conv1_bn_relu = tf.nn.relu(conv1_bn)

            pool1 = tf.layers.max_pooling2d(inputs=conv1_bn_relu, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                                     padding="same", activation=None)

            conv2_bn = tf.layers.batch_normalization(inputs=conv2, axis=-1, momentum=0.9, epsilon=0.001, center=True,
                                                     scale=True, training=self.__is_training, name='conv2_bn')
            conv2_bn_relu = tf.nn.relu(conv2_bn)

            pool2 = tf.layers.max_pooling2d(inputs=conv2_bn_relu, pool_size=[2, 2], strides=2)

            conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5],
                                     padding="same", activation=None)

            conv3_bn = tf.layers.batch_normalization(inputs=conv3, axis=-1, momentum=0.9, epsilon=0.001, center=True,
                                                     scale=True, training=self.__is_training, name='conv3_bn')
            conv3_bn_relu = tf.nn.relu(conv3_bn)

            pool3 = tf.layers.max_pooling2d(inputs=conv3_bn_relu, pool_size=[2, 2], strides=2)

            #print(pool3)
            pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 32])
            # Other way to calculate the flatten version
            #pool3_flat_v2 = tf.reshape(pool3, [-1, pool3.get_shape().as_list()[1] * pool3.get_shape().as_list()[2] *
                                               #pool3.get_shape().as_list()[3]])

            # FC layers
            FC1 = tf.layers.dense(inputs=pool3_flat, units=128, activation=tf.nn.relu)
            dropout_1 = tf.layers.dropout(inputs=FC1, rate=0.5, training=self.__is_training)
            FC2 = tf.layers.dense(inputs=dropout_1, units=64, activation=tf.nn.relu)
            #FC2 = tf.layers.dense(inputs=FC1, units=64, activation=tf.nn.relu)
            dropout_2 = tf.layers.dropout(inputs=FC2, rate=0.5, training=self.__is_training)
            self.__logits = tf.layers.dense(inputs=dropout_2, units=10)
            #self.__logits = tf.layers.dense(inputs=FC2, units=10)

            with tf.name_scope("loss_func") as scope:
                self.__loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.__logits,
                                                                                     labels=self.__y_))
                self.__loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.__logits,
                                                                                     labels=self.__y_))
                # Add loss to tensorboard
                #tf.summary.scalar("loss_train", self.__loss)
                #tf.summary.scalar("loss_val", self.__loss_val)
                self.__train_summary = tf.summary.scalar("loss_train", self.__loss)
                self.__val_summary = tf.summary.scalar("loss_val", self.__loss_val)

            # Get ops to update moving_mean and moving_variance for batch_norm
            # Reference: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.name_scope("optimizer") as scope:
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = 1e-3
                # decay every 10000 steps with a base of 0.96
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                           1000, 0.9, staircase=True)

                with tf.control_dependencies(update_ops):
                    self.__train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.__loss,
                                                                                       global_step=global_step)
                tf.summary.scalar("learning_rate", learning_rate)
                tf.summary.scalar("global_step", global_step)

            # Merge op for tensorboard
            self.__merged_summary_op = tf.summary.merge_all()

            # Build graph
            init = tf.global_variables_initializer()

            # Saver for checkpoints
            self.__saver = tf.train.Saver(max_to_keep=None)

            # Avoid allocating the whole memory
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.__session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            # Configure summary to output at given directory
            self.__writer = tf.summary.FileWriter("./logs/cifar10", self.__session.graph)
            self.__session.run(init)

    def train(self, save_dir='./save', batch_size=500):
        # Use keras to load the complete cifar dataset on memory (Not scalable)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        # Using Tensorflow data Api to handle batches
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset_train = dataset_train.shuffle(buffer_size=10000)
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.batch(batch_size)
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset_test = dataset_test.repeat()
        dataset_test = dataset_test.batch(batch_size)

        # Create an iterator
        iter_train = dataset_train.make_one_shot_iterator()
        iter_train_op = iter_train.get_next()
        iter_test = dataset_test.make_one_shot_iterator()
        iter_test_op = iter_test.get_next()

        # Build model graph
        self.build_graph()

        # Train Loop
        for i in range(20000):
            # Use CPU to get a batch of train data
            with tf.device('/cpu:0'):
                batch_train = self.__session.run([iter_train_op])
                batch_x_train, batch_y_train = batch_train[0]
            # Print loss from time to time
            if i % 100 == 0:
                # Use CPU to get a batch of test data
                with tf.device('/cpu:0'):
                    batch_test = self.__session.run([iter_test_op])
                    batch_x_test, batch_y_test = batch_test[0]

                loss_train, summary_1 = self.__session.run([self.__loss, self.__merged_summary_op],
                                                       feed_dict={self.__x_: batch_x_train,
                                                                  self.__y_: batch_y_train, self.__is_training: True})

                loss_val, summary_2 = self.__session.run([self.__loss_val, self.__val_summary],
                                                         feed_dict={self.__x_: batch_x_test,
                                                                    self.__y_: batch_y_test, self.__is_training: False})
                print("Loss Train: {0} Loss Val: {1}".format(loss_train, loss_val))
                # Write to tensorboard summary
                self.__writer.add_summary(summary_1, i)
                self.__writer.add_summary(summary_2, i)

            # Execute train op
            self.__train_step.run(session=self.__session, feed_dict={
                self.__x_: batch_x_train, self.__y_: batch_y_train, self.__is_training: True})

        # Save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_path = os.path.join(save_dir, "model")
        filename = self.__saver.save(self.__session, checkpoint_path)
        print("Model saved in file: %s" % filename)

    @property
    def output(self):
        return self.__logits

    @property
    def num_parameters(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

if __name__ == '__main__':
  fire.Fire(Train)
