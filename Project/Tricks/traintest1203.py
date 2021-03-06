"""
1203  对于训练代码进行测试

      加入不同的功能模块，然后对于训练代码进行分析

1204  在train1117的基础上进行修改
      加入微信推送和异常处理的内容，监控训练
      tensorboard倒是可以运行，但是不显示想要的图，还需要再改进一下


tensorboard usage

Python D:\software\Anaconda\envs\python36\Lib\site-packages/tensorboard\main.py
--logdir=E:\Files\Repositories\kerasYolov4/test\logs/000



"""



import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

from yolo3.model1117 import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data

import json

import requests
import traceback

sckey = 'SCU131802Tef40bc6617c6e29c898cfdc99dbcbcc55fc655fa91537'

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 按照比例分配GPU内存
# config.gpu_options.allow_growth=True  #动态分配GPU内存
K.set_session(tf.Session(config=config))


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def _main():
    annotation_path = 'traintest.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors6.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (512, 512)  # multiple of 32, hw

    model = create_model(input_shape, anchors, num_classes, freeze_body=True)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    stage1_epochs = 10
    stage2_epochs = 50
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 64
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history1 = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=stage1_epochs,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

        np.savez('trainHistory1.npz', history=history1.history)
        with open('trainHistory1.json', 'w') as f:
            json.dump(history1.history, f, cls=MyEncoder)  # 编码json文件
        print('Setp1 done! Save history to trainHistory1.json successfully!')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 8  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        history2 = model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=stage1_epochs + stage2_epochs,
            initial_epoch=stage1_epochs,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

        np.savez('trainHistory2.npz', history=history2.history)
        with open('trainHistory2.json', 'w') as f:
            json.dump(history2.history, f, cls=MyEncoder)
        print('Setp2 done! Save history to trainHistory2.json successfully!')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, freeze_body=True):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 16, 1: 8}[l], w // {0: 16, 1: 8}[l], num_anchors // 2, num_classes + 5)) for l in
              range(2)]

    model_body = yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if freeze_body:
        num = len(model_body.layers) - 2

        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': 0.1})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    print('The model has {} layers .'.format(len(model_body.layers)))
    # plot_model(model,to_file='model_data/model.png',show_shapes=True,show_layer_names=True) #储存网络结构
    model.summary()  # 打印网络

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape,
                                         random=True)  # random对于数据进行增强了，需要考虑下是不是还需要这么做
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    try:
        _main()
        url = 'https://sc.ftqq.com/%s.send?text=train1202&desp=训练成功！' % sckey
        requests.get(url)
    except Exception as e:
        print('异常类型：',repr(e))
        traceback.print_exc()
        url = 'https://sc.ftqq.com/%s.send?text=train1202&desp=训练失败！' % sckey
        requests.get(url)

