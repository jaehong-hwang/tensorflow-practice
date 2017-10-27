# -*- coding: utf-8 -*-
# Linear regression 1
# GradientDescentOptimizer라는 매직을 통해 cost를 최소화한다
# ===

import tensorflow as tf

# X, Y data set
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
# reduce_mea => 평균 함수라고 이해.
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
#Initialize global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the Line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W), sess.run(b)
