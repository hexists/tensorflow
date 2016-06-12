#!/usr/bin/env python2.7

import tensorflow as tf
import numpy as np
import input_data
import random
import math

def xavier_init(n_inputs, n_outputs, uniform=True):
    # ref : http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
             Understanding the difficulty of training deep feedforward neural
             networks. International conference on artificial intelligence and
             statistics.
    Args:
      n_inputs: The number of input nodes into each output.
      n_outputs: The number of output nodes for each input.
      uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
      An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 256], initializer=xavier_init(256, 256))
W4 = tf.get_variable("W4", shape=[256, 256], initializer=xavier_init(256, 256))
W5 = tf.get_variable("W5", shape=[256, 10], initializer=xavier_init(256, 10))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([10]))

dropout_rate = tf.placeholder("float")

_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4, W5), B5)

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
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7}) / total_batch
        if epoch % display_step == 0:
            print 'Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost)
    print 'Optimization Finished!'
    
    # Evaluation
    print '----------------------------'
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print 'Accuracy:', accuracy.eval({X: mnist.test.images, Y:mnist.test.labels, dropout_rate: 1})
