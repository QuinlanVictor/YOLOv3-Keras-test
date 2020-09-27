"""
20200927
测试设计出来的网络结构，基于yolov3对于肺结节检测这一任务进行改进，减少darknet53的单元数量，替换骨干网络的网络模块，
替换激活函数，在骨干网络后加入spp，对于检测网络进行修改，将三个尺度的输出变为一个尺度是输出，同时对于损失函数做修改，
替换中心坐标xy和宽高wh的损失函数，变为ciou_loss

在研究中继续改进，可能会有所改变。


"""
from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D,Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


"""网络结构部分"""
class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
