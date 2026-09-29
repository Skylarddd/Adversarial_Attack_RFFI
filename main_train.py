
import numpy as np
import tensorflow as tf
from tensorflow import keras
from deep_learning_models import GRU_model2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam
from load_dataset import LoadDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import signal_representations as sr
from augmentation import awgn

import matplotlib

matplotlib.use('TkAgg')
# %%
'''Load dataset and add noise'''
tx_range = np.arange(30, 40, dtype=int)
pkt_range = range(0, 900)

file_list = [
                './dataset/train/usrp_20220828train_2.h5',
                './dataset/train/usrp_20220902train_2.h5',
                './dataset/train/usrp_20220907train_2.h5',
                './dataset/train/usrp_20220915train_2.h5',
            ]


LoadDatasetObj = LoadDataset()

[data, label, _] = LoadDatasetObj.load_multiple_rx_data(file_list,
                                                        tx_range,
                                                        pkt_range)

label = label - tx_range[0]
num_classes = len(tx_range)

'''Data augmentation block'''
snr_range = np.arange(10, 60)  # define SNR range
data = awgn(data, snr_range)
# %%
'''Signal representation block'''
data = sr.dspectrogram(data, win_len=256, crop_ratio=0.3)

# %%
'''Data generate - besides CNN'''
data = data[:, :, :, 0]
data = data.transpose(0, 2, 1)
# %%
'''Define the neural network'''
model = GRU_model2(num_classes, embed_dim=data.shape[2])

patience = 20
early_stop = EarlyStopping('val_loss', min_delta=0, patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
callbacks = [early_stop, reduce_lr]

label = to_categorical(label)

'''Shuffle data and label'''
index = np.arange(len(label))
np.random.shuffle(index)
data = data[index]
label = label[index]

opt = Adam(learning_rate=1e-4)

model.compile(loss=['categorical_crossentropy'], optimizer=opt)

history = model.fit(data,
                    label,
                    epochs=500,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=1,
                    batch_size=32,
                    callbacks=callbacks)


tf.keras.models.save_model(model, 'GRU_model2.h5')
