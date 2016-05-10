#!/data1/users/daniellee/tensorflow/bin/python2.7
# -*- coding: utf-8 -*-

import tensorflow as tf

hello = tf.constant('hello, TensorFlow!')

sess = tf.Session()

print sess.run(hello)
