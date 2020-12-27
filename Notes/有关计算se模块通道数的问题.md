#### 20201227

看一下有关se模块里计算通道数的方法，使用的是如下代码进行测试


        import keras
        from keras.layers import *

        x = Input((224,224,3))
        x = Conv2D(filters=128,kernel_size=3,padding='same')(x)
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = x._keras_shape[channel_axis]
        print(channel)
        
可以计算得到channel
