#!/data1/users/daniellee/tensorflow/bin/python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('./train.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
print x_data
x_data = np.transpose(xy[0:-1])
print x_data
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

'''
# accuracy = 0.5
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# Hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))
'''

# NN
W1 = tf.Variable(tf.random_uniform([2, 4], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

# Cost Function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.5) # Learing rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for step in xrange(2000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2)
    
    # Test
    print '----------------------------'
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    
    # Calculate
    print '----------------------------'
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print 'Accuracy:', accuracy.eval({X:x_data, Y:y_data})
