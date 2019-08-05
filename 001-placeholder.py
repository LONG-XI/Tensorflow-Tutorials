#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:04:16 2019

@author: xilong
"""

"""
Dependencies:
tensorflow: 1.1.0
"""
import tensorflow as tf

# define x1 and y1 with placeholder
# placeholder(dtype: type of data, shape: dimension of data, name: give a name to this x1 )
# shape: dimension of data. such as the dimension of image tends to be [batch, height, width, channels]
# the shape of a matrix is [row, column]， it can be [None, 2] or [3, None] or None
x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
z1 = x1 + y1

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 3])
y2 = tf.placeholder(dtype=tf.float32, shape=[3, 2])
# tf.matmul(x2, y2): matrix x2 times matrix y2. it must meet the requirements of matrix multiplication. 
# one of the requirements of matrix multiplication: the coloumn of x2 has to be equal to the row of y2.
z2 = tf.matmul(x2, y2)


with tf.Session() as sess:
    # when only one operation to run
    z1_value = sess.run(z1, feed_dict={x1: 2, y1: 4})

    # when run multiple operations
    z1_value, z2_value = sess.run(
        [z1, z2],       # run them together
        feed_dict={
            x1: 2, y1: 4,
            x2: [[2,2,3], [2,1,4]], y2: [[3, 3], [2, 3], [4, 3]]
        })
    print(z1_value)
    print(z2_value)