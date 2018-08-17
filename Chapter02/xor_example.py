# References
# https://github.com/aymericdamien/TensorFlow-Examples
import tensorflow as tf
# XOR dataset
XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0], [1], [1], [0]]

num_input = 2
num_classes = 1

# Define model I/O (Placeholders are used to send/get information from graph)
x_ = tf.placeholder("float", shape=[None, num_input], name='X')
y_ = tf.placeholder("float", shape=[None, num_classes], name='Y')

# Model structure
H1 = tf.layers.dense(inputs=x_, units=4, activation=tf.nn.sigmoid)
H2 = tf.layers.dense(inputs=H1, units=8, activation=tf.nn.sigmoid)
H_OUT = tf.layers.dense(inputs=H2, units=num_classes, activation=tf.nn.sigmoid)

# Define cost function
with tf.name_scope("cost") as scope:
    cost = tf.losses.log_loss( labels=y_, predictions=H_OUT)
    # Add loss to tensorboard
    tf.summary.scalar("log_loss", cost)

# Define training ops
with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

merged_summary_op = tf.summary.merge_all()

# Initialize variables(weights) and session
init = tf.global_variables_initializer()
sess = tf.Session()
# Configure summary to output at given directory
writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)
sess.run(init)

# Train loop
for step in range(10000):
    # Run train_step and merge_summary_op
    _, summary = sess.run([train_step, merged_summary_op], feed_dict={x_: XOR_X, y_: XOR_Y})
    if step % 1000 == 0:
        print("Step/Epoch: {}, Loss: {}".format(step, sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y})))
        # Write to tensorboard summary
        writer.add_summary(summary, step)
