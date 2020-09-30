"""
date：0930
作为spp的作用，起到增加感受野的作用，ASPP和RFB和spp操作比较一致，效果会不会更好呢，这里尝试构建RFB模块

"""



from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import UpSampling2D
from keras.layers import Concatenate, Add
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
    x3 = RFBConv2D_BN_relu(num_filters, (3, 3))(x3)
    x3 = RFBConv2D_BN(num_filters, (5, 5), dilation_rate=5)(x3)

    out1 = Concatenate()([x1,x2,x3])
    out1 = RFBConv2D_BN(num_filters,(1,1))(out1)
    x = RFBConv2D_BN(num_filters,(1,1))(x)
    out = Add()([out1,x])
    out = Activation("relu")(out)

    return out
