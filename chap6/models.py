import tensorflow as tf

class CAE_CNN(object):
    def __init__(self, img_size = 28, latent_size=20):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='IMAGE_IN')
        self.__x_image = tf.reshape(self.__x, [-1, img_size, img_size, 1])

        with tf.name_scope('ENCODER'):
            ##### ENCODER
            # CONV1: Input 28x28x1 after CONV 5x5 P:2 S:2 H_out: 1 + (28+4-5)/2 = 14, W_out= 1 + (28+4-5)/2 = 14
            self.__conv1_act = tf.layers.conv2d(inputs=self.__x_image, strides=(2, 2),
                                                filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

            # CONV2: Input 14x14x16 after CONV 5x5 P:0 S:2 H_out: 1 + (14+4-5)/2 = 7, W_out= 1 + (14+4-5)/2 = 7
            self.__conv2_act = tf.layers.conv2d(inputs=self.__conv1_act, strides=(2, 2),
                                                filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        with tf.name_scope('LATENT'):
            # Reshape: Input 7x7x32 after [7x7x32]
            self.__enc_out = tf.reshape(self.__conv2_act, [tf.shape(self.__x)[0], 7 * 7 * 32])
            self.__guessed_z = tf.layers.dense(inputs=self.__enc_out,
                                               units=latent_size, activation=None, name="latent_var")
            tf.summary.histogram("latent", self.__guessed_z)


        with tf.name_scope('DECODER'):
            ##### DECODER (At this point we have 1x18x64
            self.__z_develop = tf.layers.dense(inputs=self.__guessed_z,
                                               units=7 * 7 * 32, activation=None, name="z_matrix")
            self.__z_develop_act = tf.nn.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 7, 7, 32]))

            # DECONV1
            self.__conv_t2_out_act = tf.layers.conv2d_transpose(inputs=self.__z_develop_act,
                                                                strides=(2, 2), kernel_size=[5, 5], filters=16,
                                                                padding="same", activation=tf.nn.relu)

            # DECONV2
            # Model output
            self.__y = tf.layers.conv2d_transpose(inputs=self.__conv_t2_out_act,
                                                                strides=(2, 2), kernel_size=[5, 5], filters=1,
                                                                padding="same", activation=tf.nn.sigmoid)

            # We want the output flat for using on the loss
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], 28 * 28])


    @property
    def output(self):
        return self.__y

    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input(self):
        return self.__x

    @property
    def image_in(self):
        return self.__x_image


class VAE_CNN(object):
    def __init__(self, img_size=28, latent_size=20):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='IMAGE_IN')
        self.__x_image = tf.reshape(self.__x, [-1, img_size, img_size, 1])

        with tf.name_scope('ENCODER'):
            ##### ENCODER
            # CONV1: Input 28x28x1 after CONV 5x5 P:2 S:2 H_out: 1 + (28+4-5)/2 = 14, W_out= 1 + (28+4-5)/2 = 14
            self.__conv1_act = tf.layers.conv2d(inputs=self.__x_image, strides=(2, 2),
                                                filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

            # CONV2: Input 14x14x16 after CONV 5x5 P:0 S:2 H_out: 1 + (14+4-5)/2 = 7, W_out= 1 + (14+4-5)/2 = 7
            self.__conv2_act = tf.layers.conv2d(inputs=self.__conv1_act, strides=(2, 2),
                                                filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        with tf.name_scope('LATENT'):
            # Reshape: Input 7x7x32 after [7x7x32]
            self.__enc_out = tf.reshape(self.__conv2_act, [tf.shape(self.__x)[0], 7 * 7 * 32])

            # Add linear ops for mean and variance
            self.__w_mean = tf.layers.dense(inputs=self.__enc_out,
                                            units=latent_size, activation=None, name="w_mean")
            self.__w_stddev = tf.layers.dense(inputs=self.__enc_out,
                                              units=latent_size, activation=None, name="w_stddev")

            # Generate normal distribution with dimensions [B, latent_size]
            self.__samples = tf.random_normal([tf.shape(self.__x)[0], latent_size], 0, 1, dtype=tf.float32)

            self.__guessed_z = self.__w_mean + (self.__w_stddev * self.__samples)
            tf.summary.histogram("latent_sample", self.__guessed_z)

        with tf.name_scope('DECODER'):
            ##### DECODER (At this point we have 1x18x64
            # Linear layer
            self.__z_develop = tf.layers.dense(inputs=self.__guessed_z,
                                            units=7 * 7 * 32, activation=None, name="z_matrix")
            self.__z_develop_act = tf.nn.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 7, 7, 32]))

            # DECONV1
            self.__conv_t2_out_act = tf.layers.conv2d_transpose(inputs=self.__z_develop_act,
                                                                strides=(2, 2), kernel_size=[5, 5], filters=16,
                                                                padding="same", activation=tf.nn.relu)

            # DECONV2
            # Model output
            self.__y = tf.layers.conv2d_transpose(inputs=self.__conv_t2_out_act,
                                                  strides=(2, 2), kernel_size=[5, 5], filters=1,
                                                  padding="same", activation=tf.nn.sigmoid)

            # Model output
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], 28 * 28])


    @property
    def output(self):
        return self.__y

    @property
    def z_mean(self):
        return self.__w_mean

    @property
    def z_stddev(self):
        return self.__w_stddev

    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input(self):
        return self.__x

    @property
    def image_in(self):
        return self.__x_image


class VAE_CNN_GEN(object):
    def __init__(self, img_size=28, latent_size=20):
        self.__x = tf.placeholder(tf.float32, shape=[None, latent_size], name='LATENT_IN')

        with tf.name_scope('DECODER'):
            # Linear layer
            self.__z_develop = tf.layers.dense(inputs=self.__x,
                                               units=7 * 7 * 32, activation=None, name="z_matrix")
            self.__z_develop_act = tf.nn.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 7, 7, 32]))

            # DECONV1
            self.__conv_t2_out_act = tf.layers.conv2d_transpose(inputs=self.__z_develop_act,
                                                                strides=(2, 2), kernel_size=[5, 5], filters=16,
                                                                padding="same", activation=tf.nn.relu)

            # DECONV2
            # Model output
            self.__y = tf.layers.conv2d_transpose(inputs=self.__conv_t2_out_act,
                                                  strides=(2, 2), kernel_size=[5, 5], filters=1,
                                                  padding="same", activation=tf.nn.sigmoid)

    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        return self.__x
