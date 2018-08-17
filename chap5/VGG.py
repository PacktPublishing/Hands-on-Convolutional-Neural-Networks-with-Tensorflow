   def build_graph(self): 

   self.__x_ = tf.placeholder("float", shape=[None, 224, 224, 3], name='X') 

   self.__y_ = tf.placeholder("float", shape=[None, 1000], name='Y') 

   

   with tf.name_scope("model") as scope: 

       conv1_1 = tf.layers.conv2d(inputs=self.__x_, filters=64, kernel_size=[3, 3], 

                                padding="same", activation=tf.nn.relu) 

       conv2_1 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

 

       pool1 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=2) 

 

       conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], 

                                padding="same", activation=tf.nn.relu) 

       conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

       pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2) 

      conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], 

                                padding="same", activation=tf.nn.relu) 

       conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

       conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

 

       pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2) 

 

       conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

       conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

       conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

 

       pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2) 

 

       conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

       conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

       conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3, 3], 

                                  padding="same", activation=tf.nn.relu) 

 

       pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2) 

 

       pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512]) 

       # FC Layers (can be removed) 

       fc6 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu) 

       fc7 = tf.layers.dense(inputs=fc6, units=4096, activation=tf.nn.relu) 

       # Imagenet has 1000 classes 

       fc8 = tf.layers.dense(inputs=fc7, units=1000) 

       self.predictions = tf.nn.softmax(self.fc8, name='predictions') 