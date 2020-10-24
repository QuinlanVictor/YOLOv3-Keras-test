2020.10.15

之前一直使用了限制GPU内存的代码还有打印网络结构的代码

打印：

    from keras.utils import plot_model
    
    plot_model(model,to_file='model_data/model.png',show_shapes=True,show_layer_names=True) #储存网络结构
    
    model.summary()#打印网络

限制GPU内存：

    import tensorflow as tf
    
    import keras.backend as K
    
    config = tf.ConfigProto()
    
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 #按照比例分配GPU内存
    
    #config.gpu_options.allow_growth=True  #动态分配GPU内存
    
    K.set_session(tf.Session(config=config))
