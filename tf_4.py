#coding:utf-8
#调节警告级别
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


X_SIZE = 300
BATCH_SIZE = 30
STEPS = 40000
PRINT_STEP = 2000
SEED = 2

#输入数据与标准值
rdm = np.random.RandomState(SEED)
X = rdm.randn(X_SIZE, 2)

Y_ = [[int(x0 * x0 + x1 * x1 < 2)] for (x0, x1) in X]
Y_c = [['red' if y[0] else 'blue'] for y in Y_] 

print X
print Y_
print Y_c

plt.scatter(X[:, 0], X[:, 1], c = np.squeeze(Y_c))
plt.show()

#前向传播
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape, seed = 1))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape = shape))
    return b

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2

#损失函数
loss_mse = tf.reduce_mean(tf.square(y_ - y))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

#反向传播-不含正则化
train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss_mse)
#train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        start = i * BATCH_SIZE % X_SIZE
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict = {x: X[start: end], y_: Y_[start: end]})
        
        if(i % PRINT_STEP == 0):
            loss_mse_v = sess.run(loss_mse, feed_dict = {x: X, y_: Y_})
            print("After %d steps, loss is: %f" % (i, loss_mse_v))

    #绘制等高线图
    xx, yy = np.mgrid[-3: 3: 0.01, -3: 3: 0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict = {x: grid})
    print "\nxx:\n",xx, "\nyy:\n", yy, "\ngrid:\n", grid, "\nprobs:\n", probs, "\n"
    probs = probs.reshape(xx.shape)

    plt.contour(xx, yy, probs, levels = [0.5])
    plt.scatter(X[:, 0], X[:, 1], c = np.squeeze(Y_c))
    plt.show()
            
#反向传播-含正则化
train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss_total)
#train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        start = i * BATCH_SIZE % X_SIZE
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict = {x: X[start: end], y_: Y_[start: end]})
        
        if(i % PRINT_STEP == 0):
            loss_mse_v = sess.run(loss_mse, feed_dict = {x: X, y_: Y_})
            print("After %d steps, loss is: %f" % (i, loss_mse_v))

    #绘制等高线图
    xx, yy = np.mgrid[-3: 3: 0.01, -3: 3: 0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict = {x: grid})
    print "\nxx:\n",xx, "\nyy:\n", yy, "\ngrid:\n", grid, "\nprobs:\n", probs, "\n"
    probs = probs.reshape(xx.shape)

    plt.contour(xx, yy, probs, levels = [0.5])
    plt.scatter(X[:, 0], X[:, 1], c = np.squeeze(Y_c))
    plt.show()




