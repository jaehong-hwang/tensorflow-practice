# -*- coding=utf-8 -*-
# W 에 따라 바뀌는 cost를 그래프로 그린다
# 그래프 => matplotlib.pyplot이라는 라이브러리를 사용
# ===

import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Hypothesis 함수
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
# Variables for plotting cost function
W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={
        W: feed_W,
        X: [1, 2, 3],
        Y: [1, 2, 3]
    })

    W_val.append(curr_W)
    cost_val.append(curr_cost)

# Show the cost funtcion
plt.plot(W_val, cost_val)
plt.show()
