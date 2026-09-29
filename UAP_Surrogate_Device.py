import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout, Lambda
from keras.layers.embeddings import Embedding
import keras
from tensorflow.keras.utils import to_categorical
from keras.layers import Lambda
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
import signal_representations as sr
from tensorflow.python.platform import gfile
from tensorflow.keras import backend as K
from keras import backend as K
from load_dataset import loop_load, load_single_file
from signal_representations import compute_power, eps2per
from UAPt import universal_perturbation
'''Load dataset and add noise'''

clf = tf.keras.models.load_model('GRU_Adam.h5', compile=False)
clf2 = tf.keras.models.load_model('GRU_Adam_newdevice.h5', compile=False)

per=[]
id = 0
'''Load enrollment data'''

tx_range = np.arange(30, 40, dtype=int)
tx_range2 = np.arange(40, 50, dtype=int)
classes_1 = np.arange(30, 40) - 30 + 1
classes_2 = np.arange(40, 50) - 40 + 1

[data_test, label_test] = load_single_file('./dataset/test/usrp_20221227test.h5',
                 'data',
                 'label',
                 tx_range,
                 pkt_range=range(0, 100))

label_test = label_test - tx_range[0]
label_test = to_categorical(label_test, num_classes=10)


[data_test2, label_test2] = load_single_file('./dataset/test/usrp_20230125test_newdevice.h5',
                 'data',
                 'label',
                 tx_range2,
                 pkt_range=range(0, 100))

label_test_2 = label_test2 - tx_range2[0]
label_test2 = to_categorical(label_test_2, num_classes=10)

data_test = sr.dspectrogram(data_test, win_len=256, crop_ratio=0.3)
data_test2 = sr.dspectrogram(data_test2, win_len=256, crop_ratio=0.3)
dataset_power = compute_power(data_test)
dataset2_power = compute_power(data_test2)


data_test = data_test[:, :, :, 0]
data_test = data_test.transpose(0, 2, 1)

data_test2 = data_test2[:, :, :, 0]
data_test2 = data_test2.transpose(0, 2, 1)

# target range: 0 to 9// if set target, the device will be classified as target+1
# dataset_perturbed, v = universal_perturbation(data_test, clf,target=5)
dataset_perturbed, v = universal_perturbation(data_test, clf)
dataset_perturbed_power = compute_power(dataset_perturbed)
dataset_perturbed2 = data_test2 + v

for psr in range(-40, -9):
    dataset_perturbed2, v2 = eps2per(dataset_perturbed2, data_test2, psr)
    # data_uap_adv = eps2per_uap(v, data_test, -10)
    dataset2_perturbed_power = compute_power(dataset_perturbed2)

    psr = abs(dataset2_perturbed_power - dataset2_power)/dataset2_power

    label_all_ONEHOT = clf2.predict(data_test2)
    label_all_ONEHOT = label_all_ONEHOT.argmax(axis=-1)
    label_all = to_categorical(label_all_ONEHOT, num_classes=10)
    acc_dataset2 = accuracy_score(label_test2, label_all)

    x_uap = clf2.predict(dataset_perturbed2)
    x_uap = x_uap.argmax(axis=-1)
    y_pred_uap = to_categorical(x_uap, num_classes=10)
    fool_uap = 1-accuracy_score(label_all, y_pred_uap)

    print(fool_uap*100, end=" ")

