'''把senet加入inception，理解一下思路就好'''
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, \
    Activation, Input, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

from keras.applications import InceptionV3

def build_model(nb_classes, input_shape=(256,256,3)):
    inputs_dim = Input(input_shape)
    x = InceptionV3(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(256, 256, 3),
                pooling=max)(inputs_dim)

    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=2048 // 16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=2048)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, 2048))(excitation)

    scale = multiply([x, excitation])

    x = GlobalAveragePooling2D()(scale)
    dp_1 = Dropout(0.3)(x)
    fc2 = Dense(nb_classes)(dp_1)
    fc2 = Activation('sigmoid')(fc2) #此处注意，为sigmoid函数
    model = Model(inputs=inputs_dim, outputs=fc2)
    return model

# if __name__ == '__main__':
	# model =build_model(nb_classes, input_shape=(im_size1, im_size2, channels))
	# opt = Adam(lr=2*1e-5)
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit（）
