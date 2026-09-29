import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
import matplotlib.pylab as plt
import numpy as np
import h5py
import os

'''Residual block'''
def resblock(x, kernelsize, filters, first_layer=False):
    if first_layer:
        fx = layers.Conv2D(filters, kernelsize, padding='same')(x)
        fx = layers.ReLU()(fx)
        fx = layers.Conv2D(filters, kernelsize, padding='same')(fx)
        x = layers.Conv2D(filters, 1, padding='same')(x)

        out = layers.Add()([x, fx])
        out = layers.ReLU()(out)
    else:
        fx = layers.Conv2D(filters, kernelsize, padding='same')(x)
        fx = layers.ReLU()(fx)
        fx = layers.Conv2D(filters, kernelsize, padding='same')(fx)

        out = layers.Add()([x, fx])
        out = layers.ReLU()(out)

    return out

def classification_net(datashape, num_classes):
    # datashape = datashape
    # inputs = layers.Input(shape=(np.append(datashape[1:-1],1)))
    inputs = layers.Input(shape=(datashape[1], datashape[2], datashape[3]))

    x = layers.Conv2D(32, 7, strides=2, activation='relu', padding='same')(inputs)

    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)
    x = resblock(x, 3, 64, first_layer=True)
    x = resblock(x, 3, 64)

    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=1), name='feature_layer')(x)


    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def classification_net2(datashape, num_classes):
    # datashape = datashape
    # inputs = layers.Input(shape=(np.append(datashape[1:-1],1)))
    inputs = layers.Input(shape=(datashape[1], datashape[2], datashape[3]))

    x = layers.Conv2D(32, 7, strides=2, activation='relu', padding='same')(inputs)
    x = resblock(x, 3, 64, first_layer=True)
    x = resblock(x, 3, 64)

    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=1), name='feature_layer')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model



def LSTM_model(num_classes, embed_dim=64):
    # att = layers.MultiHeadAttention(num_heads=8, key_dim=128)

    inputs = layers.Input(shape=(None, embed_dim))
    # x = layers.Masking(mask_value=0.0)(inputs)

    # x = att(inputs, inputs)

    x = layers.LSTM(256, return_sequences=True)(inputs)
    x = layers.LSTM(256, return_sequences=True)(x)

    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)

    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def LSTM_model2(num_classes, embed_dim=64):
    # att = layers.MultiHeadAttention(num_heads=8, key_dim=128)

    inputs = layers.Input(shape=(None, embed_dim))

    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(256, return_sequences=True)(x)

    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)

    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def GRU_model(num_classes, embed_dim=64):
    # att = layers.MultiHeadAttention(num_heads=8, key_dim=128)

    inputs = layers.Input(shape=(None, embed_dim))
    # x = layers.Masking(mask_value=0.0)(inputs)

    # x = att(inputs, inputs)

    x = layers.GRU(256, return_sequences=True)(inputs)
    x = layers.GRU(256, return_sequences=True)(x)

    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)

    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def GRU_model2(num_classes, embed_dim=64):
    # att = layers.MultiHeadAttention(num_heads=8, key_dim=128)

    inputs = layers.Input(shape=(None, embed_dim))
    # x = layers.Masking(mask_value=0.0)(inputs)

    # x = att(inputs, inputs)

    x = layers.GRU(128, return_sequences=True)(inputs)
    x = layers.GRU(256, return_sequences=True)(x)

    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)

    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
