# -*- coding=utf-8 -*-
# Gradient를 직접 수정해보기,
# compute_gradients, apply_gradients!
#
# 텐서플로우의 GradientDescentOptimizer 와 직접 구현한 gradient 비교 해보기
# ===

import tensorflow as tf

# tf 그래프 인풋값 ( X: 리소스, Y: 목표치 )
X = [1, 2, 3]
Y = [1, 2, 3]

# W 를 잘못된 값으로 설정한다
W = tf.Variable(5.)

# Hypothesis 함수 ( Linear model )
hypothesis = X * W

# Gradient 함수 구현
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# 비용 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# Gradient 값 가져오기
gvs = optimizer.compute_gradients(cost)
# Gradient 값 적용
apply_gradients = optimizer.apply_gradients(gvs)

# 그래프 세션 시작.
sess = tf.Session()
# 그래프 속 전역변수 초기화.
sess.run(tf.global_variables_initializer())

for step in range(21):
    print step, sess.run([gradient, W, gvs])
    sess.run(apply_gradients)
