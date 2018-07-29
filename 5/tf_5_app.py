#coding:utf-8
#调节警告级别
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import tf_5_forward as fw
import tf_5_backward as bw
from PIL import Image

THRESHOLD = 25

#恢复模型，返回预测值
def restore_model(testPicArr):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape = (None, fw.INPUT_NODE))
        y = fw.forward(x, None)
        #取第二个维度最大值索引
        preValue = tf.argmax(y, 1)
        
        #取得要恢复的滑动平均参数列表传入Saver
        variable_averages = tf.train.ExponentialMovingAverage(bw.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(bw.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue, feed_dict = {x: testPicArr})
                return preValue
            else:
                print "No checkpoint file found"
                return -1
    
    
#图片预处理 
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    for i in range(28):
        for j in range (28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < THRESHOLD:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 1
    
    print im_arr
    img_ready = im_arr.reshape([1, fw.INPUT_NODE]).astype(np.float32)
    return img_ready
    
#应用程序入口
def application():
    testNum = input("input the number of test pictures:")
    for i in range(testNum):
        testPic = raw_input("the path of test picture:")
        testPicReady = pre_pic(testPic)
        preValue = restore_model(testPicReady)
        print "The prediction number is:", preValue

def main():
	application()

if __name__ == '__main__':
	main()	
