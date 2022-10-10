from __future__ import print_function
from __future__ import absolute_import

import cv2 as cv
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, TimeDistributed, UpSampling2D
from keras.utils.data_utils import get_file
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Add

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def schedule_vgg(epoch):
    lr = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5,
          1e-5, 1e-6, 1e-6, 1e-7, 1e-7]
    return lr[epoch]

def mods_xy():
    model = Sequential()
    # conv_1
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(224, 224, 3)))#batch__input_shape=(224,224,3)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(256, 320, 3)))#batch__input_shape=(224,224,3)

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))  #180X128*64

    # conv_2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')) #90X64X128

    # conv_3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')) # 32x40X256

    # conv_4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    #
    model.add(MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', padding='same')) #32X40X512

    # conv_5
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', dilation_rate=(2, 2)))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', dilation_rate=(2, 2)))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', dilation_rate=(2, 2)))
    #
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))

    # Load weights
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    model.load_weights(weights_path)

    return model

# mods_xy().summary()

def transform_saliency(data, stateful):
    xyshift_vgg = mods_xy()
    outs = Add()([data[0], data[1]])
    # outs2 = Add()([data[0], data[2]])
    #
    # outs = Add()([outs1, outs2])

    outs = TimeDistributed(xyshift_vgg)(outs)
    # outs2 = TimeDistributed(xyshift_vgg)(data[1])
    # outs = outs + outs2
    # print(outs.shape, "changed shape")


    # concatination approach
    # residual approach
    #

    outs = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(outs)

    attention = TimeDistributed(UpSampling2D(2))(outs)
    outs = TimeDistributed(UpSampling2D(4))(outs)

    print(outs.shape, attention.shape)
    return [outs]