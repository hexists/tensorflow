#!/data1/users/daniellee/tensorflow/bin/python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, -1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, -1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# H(x) = Wx + b
hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1) # Learing rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)

    # Learns best fit is W: [1], b: [0]
    print sess.run(hypothesis, feed_dict={X:5})
    print sess.run(hypothesis, feed_dict={X:2.5})
