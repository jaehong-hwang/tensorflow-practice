# -*- coding=utf-8 -*-
# 행렬을 이용하여
# Multivariable linear regression 구현
# ===

import tensorflow as tf

'''
데이터셋을 보기쉽게 표현하면
아래 표와 같은 형식이다.

---------------------
| x           | y   |
| 73  80  75  | 152 |
| 93  88  93  | 185 |
| 89  91  90  | 180 |
| 96  98  100 | 196 |
| 73  66  70  | 142 |
---------------------
'''
x_data = [
    [73., 80., 75.],
    [93., 88., 93.],
    [89., 91., 90.],
    [96., 98., 100.],
    [73., 66., 70.]
]

y_data = [[152.], [185.], [180.], [196.], [142.]]

# shape : 인스턴스의 갯수는 n개이고, 데이터 셋은 3개이다
X = tf.placeholder(tf.float32, shape=[None, 3])

# shape : 인스턴스의 갯수는 n개이고, 데이터 셋은 1개이다
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1], name='bias'))

# X 행렬 * W 행렬
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                            feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print step, "Cost: ", cost_val, "\nPrediction:\n", hy_val

print "15, 54, 100: ", sess.run(hypothesis, feed_dict={X: [[15., 54., 100.]]})
