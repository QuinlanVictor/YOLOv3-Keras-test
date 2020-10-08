"""
date：0930
作为spp的作用，起到增加感受野的作用，ASPP和RFB和spp操作比较一致，效果会不会更好呢，这里尝试构建RFB模块

note 1007 进行一下修改，看看能否应用到网络结构中
"""

"""自己实现的RFB过程"""

from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import UpSampling2D
from keras.layers import Concatenate, Add, Multiply
from keras.layers import concatenate
from keras.layers import BatchNormalization

from keras.regularizers import l2
from yolo3.utils import compose
from functools import wraps


@wraps(Conv2D)
def RFBConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    rfb_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    rfb_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    rfb_conv_kwargs.update(kwargs)
    return Conv2D(*args, **rfb_conv_kwargs)

def RFBConv2D_BN_relu(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and ReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        RFBConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Activation("relu"))

def RFBConv2D_BN(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        RFBConv2D(*args, **no_bias_kwargs),
        BatchNormalization())


def RFB(x,num_filters):
    x1 = RFBConv2D_BN_relu(num_filters,(1,1))(x)
    x1 = RFBConv2D_BN(num_filters,(3,3),dilation_rate=1)(x1)

    x2 = RFBConv2D_BN_relu(num_filters,(1,1))(x)
    x2 = RFBConv2D_BN_relu(num_filters,(3,3))(x2)
    x2 = RFBConv2D_BN(num_filters,(3,3),dilation_rate=3)(x2)

    x3 = RFBConv2D_BN_relu(num_filters, (1, 1))(x)
    x3 = RFBConv2D_BN_relu(num_filters, (5, 5))(x3)
    x3 = RFBConv2D_BN(num_filters, (3, 3), dilation_rate=5)(x3)

    out1 = Concatenate()([x1,x2,x3])
    out1 = RFBConv2D_BN(num_filters,(1,1))(out1)
    x = RFBConv2D_BN(num_filters,(1,1))(x)
    out = Add()([out1,x])#这最后不应该是add吧，我这样编程好像通道数对不上
    out = Activation("relu")(out)


    return out





"""参考程序实现RFB"""


def conv2d_bn(x, filters, num_row, num_col, padding='same', stride=1, dilation_rate=1, relu=True):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=(stride, stride),
        padding=padding,
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    if relu:
        x = Activation("relu")(x)
    return x


def BasicRFB(x, input_filters, output_filters, stride=1, map_reduce=8):
    input_filters_div = input_filters // map_reduce

    branch_0 = conv2d_bn(x, input_filters_div * 2, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div * 2, 3, 3, relu=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, stride=stride)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, (input_filters_div // 2) * 3, 3, 3)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, stride=stride)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, dilation_rate=5, relu=False)

    branch_3 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_3 = conv2d_bn(branch_3, (input_filters_div // 2) * 3, 1, 7)
    branch_3 = conv2d_bn(branch_3, input_filters_div * 2, 7, 1, stride=stride)
    branch_3 = conv2d_bn(branch_3, input_filters_div * 2, 3, 3, dilation_rate=7, relu=False)

    out = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out


def BasicRFB_c(x, input_filters, output_filters, stride=1, map_reduce=8):
    input_filters_div = input_filters // map_reduce

    branch_0 = conv2d_bn(x, input_filters_div * 2, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div * 2, 3, 3, relu=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, stride=stride)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, (input_filters_div // 2) * 3, 1, 7)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 7, 1, stride=stride)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, dilation_rate=7, relu=False)

    out = concatenate([branch_0, branch_1, branch_2], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out


def BasicRFB_a(x, input_filters, output_filters, stride=1, map_reduce=8):
    input_filters_div = input_filters // map_reduce

    branch_0 = conv2d_bn(x, input_filters_div, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div, 3, 3, relu=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 3, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 3, 3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 1, 3)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 3, 3, dilation_rate=3, relu=False)

    branch_3 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_3 = conv2d_bn(branch_3, input_filters_div, 3, 1)
    branch_3 = conv2d_bn(branch_3, input_filters_div, 3, 3, dilation_rate=5, relu=False)

    branch_4 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_4 = conv2d_bn(branch_4, input_filters_div, 1, 3)
    branch_4 = conv2d_bn(branch_4, input_filters_div, 3, 3, dilation_rate=5, relu=False)

    branch_5 = conv2d_bn(x, input_filters_div // 2, 1, 1)
    branch_5 = conv2d_bn(branch_5, (input_filters_div // 4) * 3, 1, 3)
    branch_5 = conv2d_bn(branch_5, input_filters_div, 3, 1, stride=stride)
    branch_5 = conv2d_bn(branch_5, input_filters_div, 3, 3, dilation_rate=7, relu=False)

    branch_6 = conv2d_bn(x, input_filters_div // 2, 1, 1)
    branch_6 = conv2d_bn(branch_6, (input_filters_div // 4) * 3, 3, 1)
    branch_6 = conv2d_bn(branch_6, input_filters_div, 1, 3, stride=stride)
    branch_6 = conv2d_bn(branch_6, input_filters_div, 3, 3, dilation_rate=7, relu=False)

    out = concatenate([branch_0, branch_1, branch_2, branch_3, branch_4, branch_5, branch_6], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out


def Normalize(net):
    branch_0 = conv2d_bn(net["conv4_3"], 256, 1, 1)
    branch_1 = conv2d_bn(net['fc7'], 256, 1, 1)
    branch_1 = UpSampling2D()(branch_1)
    out = concatenate([branch_0, branch_1], axis=-1)
    out = BasicRFB_a(out, 512, 512)
    return out



