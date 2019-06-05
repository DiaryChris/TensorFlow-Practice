# coding:utf-8

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
# from scipy import misc
import imageio


# 调节警告级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MNIST_NODE = 784
RAND_NODE = 100
LAYER1_NODE = 128

BATCH_SIZE = 200
STEPS = 1000000
PIC_ROW = 6
PIC_COL = 6
PIC_MARGIN = 4

REGULARIZER = 0.00001

MODEL_SAVE_PATH = "./model/gan/"
MODEL_NAME = "gan_model"


# 权重weight初始化函数
def get_weight(shape, regularizer):

    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 偏置bias初始化函数
def get_bias(shape):

    b = tf.Variable(tf.zeros(shape))
    return b


# 辨别器待训参数
D_w1 = get_weight([MNIST_NODE, LAYER1_NODE], REGULARIZER)
D_b1 = get_bias([LAYER1_NODE])
D_w2 = get_weight([LAYER1_NODE, 1], REGULARIZER)
D_b2 = get_bias([1])
D_Variables = [D_w1, D_w2, D_b1, D_b2]


# 辨别器网络
def discriminator(x):

    x1 = tf.nn.relu(tf.matmul(x, D_w1) + D_b1)
    y = tf.nn.sigmoid(tf.matmul(x1, D_w2) + D_b2)
    return y


# 生成器待训参数
G_w1 = get_weight([RAND_NODE, LAYER1_NODE], REGULARIZER)
G_b1 = get_bias([LAYER1_NODE])
G_w2 = get_weight([LAYER1_NODE, MNIST_NODE], REGULARIZER)
G_b2 = get_bias([MNIST_NODE])
G_Variables = [G_w1, G_w2, G_b1, G_b2]


# 生成器网络
def generator(x):

    x1 = tf.nn.relu(tf.matmul(x, G_w1) + G_b1)
    y = tf.nn.sigmoid(tf.matmul(x1, G_w2) + G_b2)

    return y


# 反向传播
def backward(mnist):

    # 计算图占位
    real_x = tf.placeholder(tf.float32, shape=(None, MNIST_NODE))
    z = tf.placeholder(tf.float32, shape=(None, RAND_NODE))

    # 定义计算图
    fake_x = generator(z)
    real_y = discriminator(real_x)
    fake_y = discriminator(fake_x)

    # 全局训练步数超参数
    global_step = tf.Variable(0, trainable=False)

    # 损失函数
    D_loss = - tf.reduce_mean(tf.log(real_y) + tf.log(1.0 - fake_y))
    G_loss = - tf.reduce_mean(tf.log(fake_y))

    # 优化器
    D_train_step = tf.train.AdamOptimizer(0.0001, 0.5).minimize(D_loss, global_step=global_step, var_list=D_Variables)
    G_train_step = tf.train.AdamOptimizer(0.0001, 0.5).minimize(G_loss, global_step=global_step, var_list=G_Variables)

    # 模型保存器
    saver = tf.train.Saver()
    # 图片输出目录
    if not os.path.exists('gan_out/'):
        os.makedirs('gan_out/')

    # 开始会话
    with tf.Session() as sess:
        # 初始化所有参数
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 若有保存点，从中加载模型
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 开始训练
        for i in range(STEPS):
            # 训练鉴别器D
            real_batch_x, _ = mnist.train.next_batch(BATCH_SIZE)
            batch_z = np.random.uniform(-1, 1, (BATCH_SIZE, RAND_NODE))
            _, D_loss_, step = sess.run([D_train_step, D_loss, global_step], feed_dict={real_x: real_batch_x, z: batch_z})
            # 训练生成器G
            batch_z = np.random.uniform(-1, 1, (BATCH_SIZE, RAND_NODE))
            _, G_loss_, step = sess.run([G_train_step, G_loss, global_step], feed_dict={z: batch_z})
            # 每1000步打印损失函数值并生成图片
            if i % 1000 == 0:
                # 打印损失函数
                print("After %d training step(s), D_loss is %g, G_loss is %g." % (step, D_loss_, G_loss_))
                # 保存模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                # 生成图片
                check_imgs = sess.run(fake_x, feed_dict={z: batch_z}).reshape((BATCH_SIZE, 28, 28))[:PIC_ROW * PIC_COL]
                # 将多个生成图片绘制在一张白底图上
                imgs = np.ones(((28 + PIC_MARGIN) * PIC_ROW + PIC_MARGIN, (28 + PIC_MARGIN) * PIC_COL + PIC_MARGIN))
                for j in range(PIC_ROW * PIC_COL):
                    x_pos = PIC_MARGIN + (28 + PIC_MARGIN) * (j % PIC_ROW)
                    y_pos = PIC_MARGIN + (28 + PIC_MARGIN) * (j // PIC_ROW)
                    imgs[x_pos: x_pos + 28, y_pos: y_pos + 28] = check_imgs[j]
                imgs = imgs * 255
                imgs = imgs.astype("uint8")
                imageio.imwrite('gan_out/%s.png' % (step // 1000), imgs)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()

