#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:32:23 2019

@author: xilong
"""

import tensorflow as tf

# define two matrix constants m1 and m2
m1 = tf.constant([[3, 5]])
m2 = tf.constant([[3],
                  [8]])
# tf.matmul(m1, m2): matrix m1 times matrix m2. it must meet the requirements of matrix multiplication. 
# one of the requirements of matrix multiplication: the coloumn of m1 has to be equal to the row of m2.
dot_operation = tf.matmul(m1, m2)

print(dot_operation)  # wrong!!! no result. cannot do that!!!

# method1 use session
sess = tf.Session()
result = sess.run(dot_operation)
print(result)
sess.close()

# method2 use session, in this case, the sess can be closed automatically.
with tf.Session() as sess:
    result_ = sess.run(dot_operation)
    print(result_)