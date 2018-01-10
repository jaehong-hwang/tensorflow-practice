# -*- coding: utf-8 -*-
# Linear regression 2
# 이번엔 플레이스 홀더를 사용해본다.
#
# 플레이스 홀더의 장점은?
# => 즉흥적으로 학습데이터를 실행시킬때 넣을수 있다는것!
# ===

import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
# reduce_mea => 평균 함수라고 이해.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initialize global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the Line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
        feed_dict={X: [1, 2, 3, 4, 5],
                    Y: [5, 8, 11, 14, 17]})

    if step % 20 == 0:
        print step, cost_val, W_val, b_val

# 학습된 Hypothesis 함수로 테스트
print 'H(6) =', sess.run(hypothesis, feed_dict={X: [6]})
print 'H(1.5) =', sess.run(hypothesis, feed_dict={X: [1.5]})
print 'H(2.7) =', sess.run(hypothesis, feed_dict={X: [2.7, 4.2]})
