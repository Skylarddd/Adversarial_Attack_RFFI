import numpy as np
import h5py
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
import random
from scipy import signal
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from natsort import natsorted
import itertools


# In[]
def convert2complex(data):
    num_row = data.shape[0]
    num_col = data.shape[1]
    data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

    data_complex = data[:, :round(num_col / 2)] + 1j * data[:, round(num_col / 2):]

    return data_complex


def load_single_file(filename, dataset_name, labelset_name, dev_range, pkt_range):
    f = h5py.File(filename, 'r')
    label = f[labelset_name][:]
    label = label.astype(int)
    label = np.transpose(label)
    label = label - 1

    # cfo = f['CFO'][:]

    sample_index_list = []
    # label_selected = np.zeros([len(pkt_range)*len(dev_range),1])
    for dev_idx in dev_range:
        sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)

    data = f[dataset_name][sample_index_list]
    data = convert2complex(data)

    label = label[sample_index_list]

    f.close()
    return data, label


def loop_load(folder_name,
              rx_range,
              dataset_name,
              labelset_name,
              tx_range,
              pkt_range):
    file_list = os.listdir(folder_name)
    file_list = natsorted(file_list, key=lambda y: y.lower())
    file_list = [file_list[i] for i in rx_range]

    data = []
    label = []
    for file_idx in range(len(rx_range)):
        filename = file_list[file_idx]
        filename = folder_name + filename
        [data_rx, label_rx] = load_single_file(filename, dataset_name, labelset_name, tx_range, pkt_range)
        data.extend(data_rx)
        label.extend(label_rx)

    data = np.array(data)
    label = np.array(label)
    return data, label


def add_noise(data, snr_range):
    for pktIdx in range(data.shape[0]):
        SNRdB = random.uniform(snr_range[0], snr_range[-1])
        data[pktIdx] = awgn(data[pktIdx], SNRdB, L=1)
    return data


def awgn(s, SNRdB, L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    N0 = P / gamma  # Find the noise spectral density
    if isrealobj(s):  # check if input is real/complex object type
        n = sqrt(N0 / 2) * standard_normal(s.shape)  # computed noise
    else:
        n = sqrt(N0 / 2) * (standard_normal(s.shape) + 1j * standard_normal(s.shape))
    r = s + n  # received signal
    return r


def plt_sns_target(label_test, y_pred, accuracy, target, eps, type):
    conf_mat_fgm = confusion_matrix(label_test, y_pred)
    classes = np.arange(30, 40) - 30 + 1
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat_fgm, annot=True, annot_kws={"size": 15},
                fmt='d', cmap='Blues',
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title("Accuracy=%0.4f" % accuracy + ", Target=%d" % target + ", eps=%0.4f" % eps)
    plt.savefig("Result_{}Target_{}.pdf".format(type, target), bbox_inches='tight')
    return plt.savefig("Result_{}Target_{}.pdf".format(type, target), bbox_inches='tight')


def plt_sns_nontarget(label_test, y_pred, accuracy, eps, type):
    conf_mat_fgm = confusion_matrix(label_test, y_pred)
    classes = np.arange(30, 40) - 30 + 1
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat_fgm, annot=True, annot_kws={"size": 15},
                fmt='d', cmap='Blues',
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title("Accuracy=%0.4f" % accuracy + ", eps=%0.4f" % eps)
    plt.savefig("Result_{}.png".format(type), bbox_inches='tight')
    return plt.savefig("Result_{}.png".format(type), bbox_inches='tight')


def cal_psr(perturbation_set, signal_set):
    power_signal = sum(sum(abs(signal_set) ** 2))
    power_perturbation = sum(sum(abs(perturbation_set) ** 2))
    psr = power_perturbation / power_signal
    psrdB = 10 * math.log10(psr)
    return (psr, psrdB)


def plt_success_attack_rate_overall(target_set, acc_target_all):
    plt.figure(figsize=(10, 10))
    plt.bar(target_set, acc_target_all, label='Classified as the target')
    plt.title("Success Attack Rate(Target)", fontsize=16)
    plt.xlabel("Target", fontsize=10)
    plt.ylabel("Success Attack Rate", fontsize=10)
    plt.axis([0, 11, 0, 1.0])
    my_x_ticks = np.arange(0, 11, 1)
    my_y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.legend()
    # plt.show()
    plt.savefig('Success Attack Rate of Each Target.png', bbox_inches='tight')
    return plt.savefig('Success Attack Rate of Each Target.png', bbox_inches='tight')


def plot_confusion_matrix(cm, classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : calculate the value in confusion matrix
    - classes : Per row per column in confusion matrix
    - normalize : True:percentage False:numbers
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.figure(figsize=(5.8, 5))
    # sns.set(font_scale=1.6)
    sns.heatmap(cm, vmin=0, vmax=1, annot=True, annot_kws={"size": 13},
                fmt='.2g', cmap='Blues',
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)

    plt.xlabel('Target label', fontsize=18)
    plt.ylabel('True label', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.savefig('classification_result.pdf', bbox_inches='tight')
    # plt.savefig("Result_{}.png".format(type), bbox_inches='tight')
    plt.close()



class LoadDataset:
    def __init__(self, ):
        self.dataset_name = 'data'
        self.labelset_name = 'label'

    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        data_complex = data[:, :round(num_col / 2)] + 1j * data[:, round(num_col / 2):]
        return data_complex

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.

        INPUT:
            FILE_PATH is the dataset path.

            DEV_RANGE specifies the loaded device range.

            PKT_RANGE specifies the loaded packets range.

        RETURN:
            DATA is the laoded complex IQ samples.

            LABLE is the true label of each received packet.
        '''

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        snr = f['SNR'][:]
        snr = np.transpose(snr)
        label_start = int(label[0]) + 1
        label_end = int(label[-1]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt / num_dev)

        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

        sample_index_list = []

        for dev_idx in dev_range:
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)

        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)

        label = label[sample_index_list]
        snr = snr[sample_index_list]

        f.close()
        return data, label, snr

    def load_multiple_rx_data(self, file_list, tx_range, pkt_range):

        num_rx = len(file_list)
        num_tx = len(tx_range)
        num_pkt = len(pkt_range)

        data = []
        tx_label = []
        rx_label = []

        for file_idx in range(num_rx):
            print('Start loading dataset ' + str(file_idx + 1))
            filename = file_list[file_idx]
            [data_temp, tx_label_temp, _] = self.load_iq_samples(filename, tx_range, pkt_range)
            rx_label_temp = np.ones(num_pkt * num_tx) * file_idx

            data.extend(data_temp)
            tx_label.extend(tx_label_temp)
            rx_label.extend(rx_label_temp)

        data = np.array(data)
        tx_label = np.array(tx_label)
        rx_label = np.array(rx_label)

        return data, tx_label, rx_label

