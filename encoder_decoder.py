# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)

learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
n_input = 784
X = tf.placeholder(tf.float32, shape=[None, n_input])

n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape))    # 以正态均值mean=0和方差stddev=0.1产生正态分布值，并且该值与mean的距离在2倍方差之内


def bias_variable(shape):
    return tf.Variable(tf.random_normal(shape=shape))


def encoder(x):
    '''
    一个有4个编码隐藏层的网络
    :param x: 输入矩阵
    :return: 中间隐藏层的编码结果
    '''
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight_variable([n_input, n_hidden_1])),
                                   bias_variable([n_hidden_1])))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight_variable([n_hidden_1, n_hidden_2])),
                                   bias_variable([n_hidden_2])))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weight_variable([n_hidden_2, n_hidden_3])),
                                   bias_variable([n_hidden_3])))
    layer_4 = tf.add(tf.matmul(layer_3, weight_variable([n_hidden_3, n_hidden_4])), bias_variable([n_hidden_4]))    # 没有激活函数，便于编码输出
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight_variable([n_hidden_4, n_hidden_3])),
                                   bias_variable([n_hidden_3])))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight_variable([n_hidden_3, n_hidden_2])),
                                   bias_variable([n_hidden_2])))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weight_variable([n_hidden_2, n_hidden_1])),
                                   bias_variable([n_hidden_1])))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weight_variable([n_hidden_1, n_input])),
                                   bias_variable([n_input])))
    return layer_4

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pre = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true-y_pre, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            if epoch % display_step == 0:
                print('epoch:', epoch+1, 'loss:', '{:.9f}'.format(c))
    print('optimizer ok')

    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    plt.colorbar()
    plt.show()
