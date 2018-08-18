import tensorflow as tf

def discriminator(x):
    with tf.variable_scope("discriminator"):
        fc1 = tf.layers.dense(inputs=x, units=256, activation=tf.nn.leaky_relu)
        fc2 = tf.layers.dense(inputs=fc1, units=256, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(inputs=fc2, units=1)    
        return logits
        
def generator(z):
    with tf.variable_scope("generator"):
        fc1 = tf.layers.dense(inputs=z, units=1024, activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
        img = tf.layers.dense(inputs=fc2, units=784, activation=tf.nn.tanh)     
        return img
        
def gan_loss(logits_real, logits_fake):    
    # Target label vectors for generator and discriminator losses.
    true_labels = tf.ones_like(logits_real)
    fake_labels = tf.zeros_like(logits_fake)

    # DISCRIMINATOR loss has 2 parts: how well it classifies real images and how well it
    # classifies fake images.
    real_image_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=true_labels)
    fake_image_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=fake_labels)

    # Combine and average losses over the batch
    discriminator_loss = tf.reduce_mean(real_image_loss + fake_image_loss) 
  
    # GENERATOR is trying to make the discriminator output 1 for all its images.
    # So we use our target label vector of ones for computing generator loss.
    generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=true_labels)
    
    # Average generator loss over the batch.
    generator_loss = tf.reduce_mean(G_loss)
    
    return discriminator_loss , generator_loss 

discriminator_solver = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
generator_solver = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)

z = tf.random_uniform(maxval=1,minval=-1,shape=[batch_size, dim])

generator_sample = generator(z)

with tf.variable_scope("") as scope:   
    logits_real = discriminator(x)
    # Re-use discriminator weights
    scope.reuse_variables()
    logits_fake = discriminator(generator_sample)
    
discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

discriminator_loss, generator_loss = gan_loss(logits_real, logits_fake)

# Training steps
discriminator_train_step = discriminator_solver.minimize(discriminator_loss, var_list=discriminator_vars )
generator_train_step = generator_solver.minimize(generator_loss , var_list=generator_vars )

"""TRAINING LOOP GOES HERE"""
