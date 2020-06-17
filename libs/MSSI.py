from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import backend as K

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, add, multiply,Conv2DTranspose,Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle

from tensorflow.keras import backend as K
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def MSSI_Model_Low():
    input_img = Input(shape=(None, None, 1))
    input_mask = Input(shape=(None, None, 1))

    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(input_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)

    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_mask = Conv2D(1, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.Ones(), trainable = False)(output_mask)
    mask_ratio = tf.keras.layers.Lambda(lambda tensor: 9 / (tensor + 1e-8))(output_mask)
    output_mask = tf.keras.layers.Lambda(lambda tensor: K.clip(tensor, 0, 1))(output_mask)
    mask_ratio = multiply([output_mask, mask_ratio])
    output_img = multiply([mask_ratio, output_img])
    output_img = Activation('relu')(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)

    model = Model(inputs = [input_img, input_mask], outputs = [output_img])
    return model


def MSSI_Model_High():
    input_img = Input(shape=(None, None, 64))
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    output_img = Activation("relu")(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)

    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)
    output_img = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)
    output_img = Activation("relu")(output_img)
    res_img = output_img
    output_img = add([res_img, input_img])
    output_img = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(output_img)

    model = Model(inputs = [input_img], outputs = [output_img])
    return model



def MSSI_Model():
    low_part = MSSI_Model_Low()
    high_part = MSSI_Model_High()

    input_img = Input(shape=(None, None, 1))
    input_mask = Input(shape=(None, None, 1))

    output_img = low_part([input_img, input_mask])
    output_img = high_part([output_img])

    res_img = multiply([output_img,1 - input_mask])
    output_img = add([res_img, input_img])

    model = Model(inputs = [input_img, input_mask], outputs = [output_img])
    return model

def MSSI_Model_parts():
    low_part = MSSI_Model_Low()
    high_part = MSSI_Model_High()

    input_img = Input(shape=(None, None, 1))
    input_mask = Input(shape=(None, None, 1))

    output_img = low_part([input_img, input_mask])
    output_img = high_part([output_img, input_mask])

    res_img = multiply([output_img,1 - input_mask])
    output_img = add([res_img, input_img])

    model = Model(inputs = [input_img, input_mask], outputs = [output_img])
    return model, low_part, high_part