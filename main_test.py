import numpy as np
import math
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Flatten, Dropout, Lambda
import keras
from keras.layers import Lambda
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import signal_representations as sr
from keras import backend as K
from sklearn.metrics import plot_confusion_matrix
from load_dataset import load_single_file


'''Load dataset and add noise'''
clf = tf.keras.models.load_model('GRU_Adam.h5', compile=False)

'''Load enrollment data'''
tx_range = np.arange(30, 40, dtype=int)

[data_test, label_test] = load_single_file('./dataset/test/usrp_20221227test.h5',
                                           'data',
                                           'label',
                                           tx_range,
                                           pkt_range=range(0, 100))

label_test = label_test - tx_range[0]
data_test = sr.dspectrogram(data_test, win_len=256, crop_ratio=0.3)

data_test = data_test[:, :, :, 0]
data_test = data_test.transpose(0, 2, 1)

pred_prob = clf.predict(data_test)
pred_label = pred_prob.argmax(axis=-1)
conf_mat = confusion_matrix(label_test, pred_label)
acc = accuracy_score(label_test, pred_label)
print('Overall accuracy = %.4f' % acc)

classes = np.arange(30, 40) - 30 + 1
plt.figure(figsize=(3.4, 3))

sns.heatmap(conf_mat, annot=True, annot_kws={"size": 6},
            fmt = 'd', cmap='Blues',
            cbar = False,
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel('Predicted label', fontsize = 9)
plt.ylabel('True label', fontsize = 9)
plt.xticks(size=8)
plt.yticks(size=8)
plt.savefig('GRU.pdf', bbox_inches='tight')
plt.close()
