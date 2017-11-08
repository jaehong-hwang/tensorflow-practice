# -*- coding=utf-8 -*-
# 파일에서 행렬 데이터를 불러오고,
# Multivariable linear regression 구현
# ===

import tensorflow as tf
import numpy as np

# csv 파일로부터 X, Y 데이터 불러오기
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_len = len(x_data[0])
y_len = len(y_data[0])

# 데이터가 정상인지 확인하는 작업
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
print(x_len, y_len)

# 학습을 위한 플레이스홀더 집합 변수 생성
# 모양별 길이는 실제 데이터의 길이로 하기위해 len함수 사용
X = tf.placeholder(tf.float32, shape=[None, x_len])
Y = tf.placeholder(tf.float32, shape=[None, y_len])

W = tf.Variable(tf.random_normal([x_len, y_len]), name='weight')
b = tf.Variable(tf.random_normal([y_len]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# 비용 함수 구현
cost = tf.reduce_mean(tf.square(hypothesis - Y));

# 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# 세션 시작
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print step, "Cost: ", cost_val, "\nPrediction:\n", hy_val

print "Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]})
print "Other score will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]})
