#coding:utf-8
#调节警告级别
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
X_SIZE = 32
STEPS = 3000
PRINT_STEP = 500
SEED = 23455

#随机生成输入矩阵
rdm = np.random.RandomState(SEED)
X = rdm.rand(X_SIZE, 2)

#正确结果列向量
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print "X:\n", X 
print "Y_:\n", Y_ 

#前向传播
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#反向传播
loss_mse = tf.reduce_mean(tf.square(y_ - y))
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss_mse)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

#生成会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print "w1:\n", sess.run(w1) 
    print "w2:\n", sess.run(w2) 
    print "\n" 

    for i in range(STEPS):
        start = (i * BATCH_SIZE % X_SIZE)
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict = {x: X[start: end], y_: Y_[start: end]})
        if i % PRINT_STEP == 0:
            total_loss = sess.run(loss_mse, feed_dict = {x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))

    print "\n" 
    print "w1:\n", sess.run(w1) 
    print "w2:\n", sess.run(w2) 

















