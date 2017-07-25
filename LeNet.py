# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def weight_variable(shape, name=None):
    with tf.name_scope('%s' % name):
        weight = tf.truncated_normal(shape=shape, mean=0, stddev=0.1, name=name)
        return tf.Variable(weight)


def bias_variable(shape, name):
    with tf.name_scope('%s' % name):
        bias = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(bias)

xs = tf.placeholder(tf.float32, shape=[None, 784])
ys = tf.placeholder(tf.float32, shape=[None, 10])
inputs_images = tf.reshape(xs, [-1, 28, 28, 1])

conv1_weight = weight_variable([5, 5, 1, 6], 'conv1_weight')
conv1_bias = bias_variable([6], 'conv1_bias')
conv_1 = tf.nn.conv2d(inputs_images, conv1_weight, [1, 1, 1, 1], padding='SAME', name='conv1')  # 28*28*6
conv1_pool = tf.nn.max_pool(tf.nn.relu(conv_1+conv1_bias), [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='conv1_pool') # 14*14*6


conv2_weight = weight_variable([5, 5, 6, 16], 'conv2_weight')
conv2_bias = bias_variable([16], 'conv2_bias')
conv_2 = tf.nn.conv2d(conv1_pool, conv2_weight, [1, 1, 1, 1], padding='VALID', name='conv2') # 10*10*16
conv2_pool = tf.nn.max_pool(tf.nn.relu(conv_2+conv2_bias), [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='conv2_pool') # 5*5*16

flatten = tf.reshape(conv2_pool, [-1, 5*5*16])

fc1_weight = weight_variable([5*5*16, 120], name='fc1_weight')
fc1_bias = bias_variable([120], 'fc1_bias')
fc1 = tf.nn.relu(tf.matmul(flatten, fc1_weight)+fc1_bias)

fc2_weight = weight_variable([120, 84], name='fc2_weight')
fc2_bias = bias_variable([84], name='fc2_bias')
fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weight) + fc2_bias)

output_weight = weight_variable([84, 10], 'output_weight')
output_bias = bias_variable([10], name='output_bias')
output = tf.nn.softmax(tf.matmul(fc2, output_weight) + output_bias, name='output')

cross_entory = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(output), reduction_indices=[1]))
train = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entory)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train, feed_dict={xs: batch_x, ys: batch_y})
        if i % 20 == 0:
            for j in range(10):
                testData = mnist.test.next_batch(50)
                prediction = sess.run(output, feed_dict={xs: testData[0]})
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(testData[1], 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print('iteration:', i, 'accuracy:', sess.run(accuracy))









