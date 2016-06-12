#!/usr/bin/env python2.7

import tensorflow as tf
import numpy as np
import input_data
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
hypothesis = tf.add(tf.matmul(L2, W3), B3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch
        if epoch % display_step == 0:
            print 'Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost)
    print 'Optimization Finished!'
    
    # Evaluation
    print '----------------------------'
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print 'Accuracy:', accuracy.eval({X: mnist.test.images, Y:mnist.test.labels})
