#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:47:21 2019

@author: xilong
"""

# In this tutorial, we import plt to plot.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
x_data = np.linspace(-6, 6, 100)     # x data, shape=(100, 1)
print(x_data)

# following are popular activation functions
y_relu = tf.nn.relu(x_data)  # relu activation function
y_sigmoid = tf.nn.sigmoid(x_data)  # sigmoid activation function
y_tanh = tf.nn.tanh(x_data)  # tanh activation function
y_softplus = tf.nn.softplus(x_data)
# y_softmax = tf.nn.softmax(x)  softmax is a special kind of activation function, it is about probability

sess = tf.Session()
y_relu, y_sigmoid, y_tanh, y_softplus = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus])

# plt to visualize these activation function
# figure(num=None, figsize=(width, heigth), dpi=None, facecolor=background color, edgecolor=edge color, frameon=True)
plt.figure(1, figsize=(8, 6))

# plt.subplot(nrows,ncols,sharex,sharey,subplot_kw,**fig_kw)
plt.subplot(2,2,1)
plt.plot(x_data, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(x_data, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(x_data, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(x_data, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
sess.close()