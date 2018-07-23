#coding:utf-8
#调节警告级别
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf_4_generate as gr
import tf_4_forward as fw

X_SIZE = 300
BATCH_SIZE = 30
STEPS = 40000
PRINT_STEP = 2000
SEED = 2
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.004
ISREGULARIZED = True

def backward():

    x = tf.placeholder(tf.float32, shape = (None, 2))
    y_ = tf.placeholder(tf.float32, shape = (None, 1))

    X, Y_, Y_c = gr.generate(X_SIZE, SEED)
    y = fw.forward(x, REGULARIZER)
    
    #指数衰减学习率
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		X_SIZE/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)

    #损失函数
    loss_mse = tf.reduce_mean(tf.square(y_ - y))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    if ISREGULARIZED:
        #反向传播-含正则化
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
    else:
        #反向传播-不含正则化
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_mse)
    

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = i * BATCH_SIZE % X_SIZE
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict = {x: X[start: end], y_: Y_[start: end]})
            
            if i % PRINT_STEP == 0:
                loss_mse_v = sess.run(loss_mse, feed_dict = {x: X, y_: Y_})
                print("After %d steps, loss is: %f" % (i, loss_mse_v))

        #绘制等高线图
        xx, yy = np.mgrid[-3: 3: 0.01, -3: 3: 0.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict = {x: grid})
        probs = probs.reshape(xx.shape)
        
        plt.contour(xx, yy, probs, levels = [0.5])
        plt.scatter(X[:, 0], X[:, 1], c = np.squeeze(Y_c))
        plt.show()
                
if __name__ == "__main__":
    backward()
    
                
