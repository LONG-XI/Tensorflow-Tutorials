#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:39:28 2019

@author: xilong
"""

import tensorflow as tf

x = tf.Variable(1)    # our first variable in the "global_variable" set

# this function is the addition function.
add_operation = tf.add(x, 6)

# assign the value of add_operation to x
update_operation = tf.assign(x, add_operation)

with tf.Session() as sess:
    # once define variables, you have to initialize them by doing this
    sess.run(tf.global_variables_initializer())
    for _ in range(7):
        sess.run(update_operation)
        print(sess.run(x))
