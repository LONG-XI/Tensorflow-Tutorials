#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:40:01 2019

@author: xilong
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


tf.set_random_seed(1)
np.random.seed(1)

# create data
# [: , np.newaxis] and [np.newaxis，：] are adding a dimension to the original array changing that array to a matirx.
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
print('x.shape is ' , x.shape)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise  
print(y)                        # shape (100, 1) + some noise
print('y.shape is ', y.shape)
# plot data ， Scatter plot
plt.scatter(x, y)
plt.show()

# define tf_x and tf_y with placeholder, if you don't know how to use placeholder, please check out 001-placeholder.py in this tutorials.
tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 20, tf.nn.relu)          # hidden layer
print('l1.shape is ',l1.get_shape())

output = tf.layers.dense(l1, 1)                     # output layer
print('output.shape is ',output.get_shape())

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)  # gradient descent
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting

for step in range(100):
    # train and net output, pred is short for prediction
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 2 == 0:
        # plot and show learning process
        plt.cla()  # clear the last line
        plt.scatter(x, y) # display the scatter plot
        plt.plot(x, pred, 'r-', lw=5) # display the new line
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})  # display the text in plane. 0.5 and 0 is the coordinates of the output information. The origin coordinates are (0,0).
        plt.pause(0.1)

plt.ioff()
plt.show()