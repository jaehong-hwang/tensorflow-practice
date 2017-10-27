# -*- coding=utf-8 -*-
# Gradient descent를 직접 구현해 본다
# 미분을 통해 기울기를 구해 비용 최소화
# ===

import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Hypothesis 함수
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimiz: Gradient descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X: [1, 5, 4], Y: [2, 10, 8]})
    print step, sess.run(cost, feed_dict={X: [1, 5, 4], Y: [2, 10, 8]}), sess.run(W)
