# -*- coding: utf-8 -*-
import tensorflow as tf

Matrix1 = tf.constant([[3, 3]])
Matrix2 = tf.constant([[2], [2]])
product = tf.matmul(Matrix1, Matrix2)

# sess = tf.Session()
# res = sess.run(product)
# print(res)
# sess.close()


with tf.Session() as sess:
    res = sess.run(product)
    print(res)


# session use with