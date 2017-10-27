# -*- coding=utf-8 -*-
# 직접 구현했던 GradientDescent를 이미 존재하는것으로 활용,
# GradientDescentOptimizer를 활용해 Minimize를 한다
# ===

import tensorflow as tf

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# set wrong model weights
W = tf.Variable(-3.0)

# Hypothesis 함수 ( Linear model )
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(21):
    print step, sess.run(W), sess.run(cost)
    sess.run(train)
