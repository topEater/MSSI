from __future__ import absolute_import, division, print_function, unicode_literals
import time

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU,Flatten
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, add, multiply, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from MSSI import *

import pickle

from tensorflow.keras import backend as K
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def make_discriminator_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2,
                     input_shape=(41,41,2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    input_img = Input(shape=(41, 41, 1))
    input_con = Input(shape=(41, 41, 1))

    img = tf.keras.layers.Concatenate(axis=-1)([input_img, input_con])
    validity = model(img)

    return Model([input_img, input_con], validity)


data = np.load("./data/terrain-41-test.npy")
# y_train = data
y_train = np.concatenate(
    (data[0:10000, :, :, :], data[0:10000, :, :, :], data[0:10000, :, :, :]))
# mask_train = np.zeros(y_train.shape)
# mask_train[0:     10000, 0:41:4, 0:41:4, 0] = 1
# mask_train[10000:  20000, 0:41:3, 0:41:3, 0] = 1
# mask_train[20000:  30000, 0:41:2, 0:41:2, 0] = 1

mask_train = np.ones(y_train.shape)
mask_train[0    :  10000, 5 :-5 , 5:-5  , 0] = 0
mask_train[10000:  20000, 10:-10, 10:-10, 0] = 0
mask_train[20000:  30000, 20:-20, 20:-20, 0] = 0

x_train = np.multiply(mask_train, y_train)

Discriminator = make_discriminator_model()
Generator = MSSI_Model()
Generator.load_weights("./model/MSSI_val_loss_best_weights.h5")

adam = Adam(lr=0.0001)
input_img = Input(shape=(41, 41, 1))
input_con = Input(shape=(41, 41, 1))
output_label = Discriminator([input_img, input_con])
DM = Model(inputs=[input_img, input_con], outputs=output_label)
DM.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# fake_image = np.load("fake_image.npy")

fake_image = Generator.predict([x_train, mask_train])
np.save("fake_image.npy", fake_image)

# temp = np.concatenate((fake_image[0:40000,:,:,:],fake_image[80000:12000,:,:,:],fake_image[160000:200000,:,:,:]))
# print(temp.shape)

print(y_train.shape)
x = np.concatenate((y_train[:, :, :, :], fake_image))
print(x.shape)
x_con = np.concatenate(
    (x_train[:, :, :, :], x_train[:, :, :, :]))
y = np.ones([2*30000, 1])
y[30000:, :] = 0
d_loss = DM.fit([x, x_con], y, shuffle=True, epochs=5)
Discriminator.save_weights("./model/dm-pretrain-void.h5")
