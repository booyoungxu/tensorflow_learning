# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # download data, if exists, use it


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 初始化单个卷积核的参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    # 以正态均值mean=0和方差stddev=0.1产生正态分布值，并且该值与mean的距离在2倍方差之内
    return tf.Variable(initial)

# 初始化卷积核上的偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 输入特征x，用卷积核进行卷积运算，strides为卷积移动步长，padding表示是否需要补充边缘对齐使其产生的图像和原图像大小一样
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 对x进行最大池化操作，ksize进行池化的范围，[batch, h, w, channel]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
x_image = tf.reshape(xs, [-1, 28, 28, 1])   # samples size channel


# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])    # 32个对于1通道的5×5卷积核
b_conv1 = bias_variable([32])   # 32个feature map代表32个神经元，每个神经元有一个偏置
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # 14x14x32

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])   # 卷积核
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 14x14x64
h_pool2 = max_pool_2x2(h_conv2)   # 7x7x64


# func1 layer
W_fcl = weight_variable([7*7*64, 1024])
b_fcl = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)
h_fcl_drop = tf.nn.dropout(h_fc1, keep_prob)

##func2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# prediction = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)
prediction = tf.maximum(tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2), 1e-30)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(50)    # 分批次最小化损失函数，每次100张图片
    sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
    if i % 50 == 0:
        for j in range(10):
            testSet = mnist.test.next_batch(50)
            print(compute_accuracy(testSet[0], testSet[1]))
    if i == 0:
        print('prediction:', sess.run(prediction, feed_dict={xs: batch_xs, keep_prob: 1}))

