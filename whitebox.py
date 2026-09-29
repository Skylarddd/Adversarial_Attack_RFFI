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
from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
import signal_representations as sr
from tensorflow.python.platform import gfile
from tensorflow.keras import backend as K
from keras import backend as K
from sklearn.metrics import plot_confusion_matrix
from load_dataset import load_single_file
from signal_representations import compute_power, eps2per
from UAPt import universal_perturbation
'''Load dataset and add noise'''

clf = tf.keras.models.load_model('GRU_Adam.h5', compile=False)

per=[]
id = 0
'''Load enrollment data'''

tx_range = np.arange(30, 40, dtype=int)
classes = np.arange(30, 40) - 30 + 1

[data_test, label_test] = load_single_file('./dataset/test/usrp_20221227test.h5',
                 'data',
                 'label',
                 tx_range,
                 pkt_range = range(0, 100))

label_test = label_test - tx_range[0]
label_test = to_categorical(label_test, num_classes=10)

data_test = sr.dspectrogram(data_test, win_len=256, crop_ratio=0.3)
dataset_power = compute_power(data_test)

data_test = data_test[:, :, :, 0]
data_test = data_test.transpose(0, 2, 1)


# target range: 0 to 9// if set target, the device will be classified as target+1
# dataset_perturbed, v = universal_perturbation(data_test, clf,target=5)
dataset_perturbed, v = universal_perturbation(data_test, clf)
dataset_perturbed_power = compute_power(dataset_perturbed)


for psr in range(-40, -9):
    dataset_perturbed, v = eps2per(dataset_perturbed, data_test, psr)
    dataset_perturbed_power = compute_power(dataset_perturbed)
    x_uap = clf.predict(dataset_perturbed)
    x_uap = x_uap.argmax(axis=-1)
    y_pred_uap = to_categorical(x_uap, num_classes=10)
    fool_uap = 1-accuracy_score(label_test, y_pred_uap)

    # print(
    #     ' fool_uap = %.4f, psr = %.4f' % (fool_uap, psr))
    print(fool_uap*100, end=" ")