# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def weight_variable(shape, name=None):
    with tf.name_scope('%s' % name):
        weight = tf.truncated_normal(shape=shape, stddev=0.1, name=name)
        return tf.Variable(weight)


def bias_variable(shape, name):
    with tf.name_scope('%s' % name):
        bias = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(bias)

xs = tf.placeholder(tf.float32, shape=[None, 784])
ys = tf.placeholder(tf.float32, shape=[None, 10])
inputs_images = tf.reshape(xs, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    conv1_weight = weight_variable([5, 5, 1, 6], 'conv1_weight')
    tf.summary.histogram('conv1_weight', conv1_weight)
    conv1_bias = bias_variable([6], 'conv1_bias')
    tf.summary.histogram('conv1_bias', conv1_bias)
    conv_1 = tf.nn.conv2d(inputs_images, conv1_weight, [1, 1, 1, 1], padding='SAME', name='conv1')  # 28*28*6
    conv1_pool = tf.nn.max_pool(tf.nn.relu(conv_1+conv1_bias), [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='conv1_pool') # 14*14*6
    tf.summary.histogram('conv1_output', conv1_pool)

with tf.name_scope('conv2'):
    conv2_weight = weight_variable([5, 5, 6, 16], 'conv2_weight')
    conv2_bias = bias_variable([16], 'conv2_bias')
    conv_2 = tf.nn.conv2d(conv1_pool, conv2_weight, [1, 1, 1, 1], padding='VALID', name='conv2') # 10*10*16
    conv2_pool = tf.nn.max_pool(tf.nn.relu(conv_2+conv2_bias), [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='conv2_pool') # 5*5*16

with tf.name_scope('conv3'):
    conv3_weight = weight_variable([5, 5, 16, 120], name='conv3_weight')
    conv3_bias = bias_variable([120], name='conv3_bias')
    conv3 = tf.nn.conv2d(conv2_pool, conv3_weight, [1, 1, 1, 1], padding='VALID', name='conv3')
    conv3_relu = tf.nn.relu(conv3+conv3_bias)

flatten = tf.reshape(conv3_relu, [-1, 120])
with tf.name_scope('fc1'):
    fc1_weight = weight_variable([120, 84], name='fc1_weight')
    fc1_bias = bias_variable([84], name='fc1_bias')
    fc1 = tf.nn.relu(tf.matmul(flatten, fc1_weight) + fc1_bias)

dropout1 = tf.placeholder(tf.float32)
fc1_dropout = tf.nn.dropout(fc1, dropout1)
with tf.name_scope('output'):
    output_weight = weight_variable([84, 10], 'output_weight')
    output_bias = bias_variable([10], name='output_bias')
    output = tf.nn.softmax(tf.matmul(fc1_dropout, output_weight)/dropout1 + output_bias, name='output')

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(output), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()  # 合并所有summary
    writer = tf.summary.FileWriter('log', sess.graph)  # 在项目目录下运行tensorboard --logdir=log 必须将框架图先写入文件
    for i in range(20000):
        batch_x, batch_y = mnist.train.next_batch(50)
        train.run(feed_dict={xs: batch_x, ys: batch_y, dropout1: 0.5})
        if i % 100 == 0:
            result = sess.run(merged, feed_dict={xs: batch_x, ys: batch_y, dropout1: 1.0})
            writer.add_summary(result, i)
            train_accuracy = accuracy.eval(feed_dict={
                xs: batch_x, ys: batch_y, dropout1: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

    for j in range(10):
        testData = mnist.test.next_batch(50)
        print('iteration:', j, 'accuracy:', sess.run(accuracy, feed_dict={xs: testData[0], ys: testData[1], dropout1: 0.8}))









