# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram('Weights', Weights)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram('bias', bias)
        Wx_plus_b = tf.matmul(inputs, Weights) + bias
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        tf.summary.histogram('outputs', output)
        return output
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, [300, 1])
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion() # lianxu
plt.show()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./', sess.graph) # tensorboard --logdir='./'
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)


