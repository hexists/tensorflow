#!/data1/users/daniellee/tensorflow/bin/python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('./train.txt', unpack=True, dtype='float32')

x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])
print '----------------------------'
print x_data
print y_data
print '----------------------------'

X = tf.placeholder('float', [None, 3])
Y = tf.placeholder('float', [None, 3])

# Set Model weights
W = tf.Variable(tf.zeros([3, 3]))

# Hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# minimum error using cross entropy
learning_rate = 0.001

# Cost Function(cross entropy)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices = 1))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

print '----------------------------'
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
print a, sess.run(tf.arg_max(a, 1))
b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
print b, sess.run(tf.arg_max(b, 1))
c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
print c, sess.run(tf.arg_max(c, 1))
all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
print all, sess.run(tf.arg_max(all, 1))
