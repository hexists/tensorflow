#!/data1/users/daniellee/tensorflow/bin/python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('./train.txt', unpack=True, dtype='float32')

x_data = xy[0:-1] 
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, -1.0))

# Hypothesis
hypothesis = tf.matmul(W, x_data)

# Cost
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1) # Learing rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(2001):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(cost), sess.run(W)
