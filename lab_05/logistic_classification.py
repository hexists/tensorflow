#!/data1/users/daniellee/tensorflow/bin/python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('./train.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]
print x_data
print y_data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# Hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# Cost Function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.1) # Learing rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

print '----------------------------'
print sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5
print sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5
print sess.run(hypothesis, feed_dict={X:[[1, 1], [4, 3], [3, 5]]}) > 0.5
