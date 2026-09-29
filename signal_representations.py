import numpy as np
import tensorflow as tf
from scipy import signal
import math
import numpy as np
import h5py
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy import signal

# In[]

def normalization(data):
    ''' Normalize the signal.'''
    s_norm = np.zeros(data.shape, dtype=complex)

    for i in range(data.shape[0]):

        sig_amplitude = np.abs(data[i])
        rms = np.sqrt(np.mean(sig_amplitude**2))
        s_norm[i] = data[i]/rms

    return s_norm


def iq_data(data):
    data = normalization(data)

    num_sample = data.shape[0]
    num_row = 2
    num_column = data.shape[1]

    data_iq = np.zeros([num_sample, num_row, num_column, 1])
    for i in range(num_sample):
        data_iq[i,0,:,0] = np.real(data[i])
        data_iq[i,1,:,0] = np.imag(data[i])

    return data_iq

def fft_results(data):
    data = normalization(data)

    num_sample = data.shape[0]
    num_row = 2
    num_column = data.shape[1]

    data = np.fft.fft(data)
    # data = np.fft.fftshift(data)

    data_fft = np.zeros([num_sample, num_row, num_column, 1])
    for i in range(num_sample):
        data_fft[i,0,:,0] = np.log10(np.abs(data[i]))
        data_fft[i,1,:,0] = np.angle(data[i])

    return data_fft



def psd(data):
    data = normalization(data)

    num_sample = data.shape[0]
    num_column = data.shape[1]


    data = np.fft.fft(data)
    data = np.fft.fftshift(data)

    data_fft = np.zeros([num_sample, num_column])
    for i in range(num_sample):
        data_fft[i,:] = np.log10(np.abs(data[i]))

    return data_fft

# def fft_results(data):
#     data = normalization(data)

#     num_sample = data.shape[0]
#     num_column = data.shape[1]

#     data = np.fft.fft(data)
#     # data = np.fft.fftshift(data)

#     data_fft = np.zeros([num_sample, num_column])
#     for i in range(num_sample):
#         data_fft[i,:] = np.log10(np.abs(data[i]))

#     return data_fft


# def spec_shift(x):
#     num_row = x.shape[0]
#     num_col = x.shape[1]

#     x_shifted = np.zeros([num_row, num_col], dtype = complex)
#     x_shifted[:round(num_row/2)] = x[round(num_row/2):]
#     x_shifted[round(num_row/2):] = x[:round(num_row/2)]

#     return x_shifted

def spec_crop(x, crop_ratio):

    num_row = x.shape[0]
    x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]

    return x_cropped

def spectrogram(data, win_len, crop_ratio):
    data = normalization(data)

    overlap = round(0.5*win_len)

    num_sample = data.shape[0]

    num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
    num_column = int(np.floor((data.shape[1]-win_len)/(win_len - overlap)) + 1)

    data_spec = np.zeros([num_sample, num_row, num_column, 1])

    for i in range(data.shape[0]):

        f, t, spec = signal.stft(data[i],
                                window='boxcar',
                                nperseg = win_len,
                                noverlap = overlap,
                                nfft = win_len,
                                return_onesided=False,
                                padded = False,
                                boundary = None)

        spec = np.fft.fftshift(spec, axes=0)
        spec = spec_crop(spec, crop_ratio)
        spec = np.log10(np.abs(spec)**2)

        data_spec[i,:,:,0] = spec

    return data_spec


def dspectrogram(data, win_len, crop_ratio):
    data = normalization(data)

    # win_len = 16
    overlap = round(0.5*win_len)

    num_sample = data.shape[0]
    # num_row = math.ceil(win_len*(1-2*crop_ratio))
    num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
    num_column = int(np.floor((data.shape[1]-win_len)/(win_len - overlap)) + 1) - 1


    data_dspec = np.zeros([num_sample, num_row, num_column, 1])
    # data_dspec = []
    for i in range(num_sample):

        dspec_amp = gen_differential_spectrogram(data[i], win_len, overlap)
        dspec_amp = spec_crop(dspec_amp, crop_ratio)
        data_dspec[i,:,:,0] = dspec_amp
        # data_dspec[i,:,:,1] = dspec_phase

    return data_dspec


def gen_differential_spectrogram(sig, win_len, overlap):
    f, t, spec = signal.stft(sig,
                            window='boxcar',
                            nperseg= win_len,
                            noverlap= overlap,
                            nfft= win_len,
                            return_onesided=False,
                            padded = False,
                            boundary = None)

    # spec = spec_shift(spec)
    spec = np.fft.fftshift(spec, axes=0)
    # spec = spec_crop(spec, crop_ratio)

    # dspec = np.zeros([spec.shape[0],spec.shape[1]-1], dtype = complex)
    # for j in range(dspec.shape[1]):
    #     dspec[:,j] = spec[:,j] / spec[:,j+1]

    dspec = spec[:,1:]/(spec[:,:-1]+1e-32)

    dspec_amp = np.log10(np.abs(dspec)**2+1e-32)
    # dspec_phase = np.angle(dspec)

    return dspec_amp


#guanxiong
def awgn(data, snr_range):
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    SNR_linear = 10**(SNRdB/10)

    data_noisy = np.empty(data.shape,dtype = "complex_")
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        # SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear[pktIdx]
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))

        if np.isnan(np.sum(n)) or np.isinf(np.sum(n)):
            data_noisy[pktIdx] = s
        else:
            data_noisy[pktIdx] = s + n
    return data_noisy



def wgn(data, per):
    per = 10**(per/10)
    pkt_num = data.shape[0]
    data_add_AWGN = np.empty(data.shape, dtype="float")
    N0 = np.empty(data.shape)
    for pktIdx in range(pkt_num):
        # ====================RNN===========================
        s = data[pktIdx]
        # data = data[:, :, :, 0]
        data_set_ind = data[pktIdx, :, :]

        N0[pktIdx] = np.power(data_set_ind, 2) * per
        # n = sqrt(N0[pktIdx]) * standard_normal(size=(64, 62, 1))
        n = sqrt(N0[pktIdx]) * standard_normal(size=(62, 104))
        # ====================CNN===========================
        # s = data[pktIdx]
        # data_set_ind = data[pktIdx, :, :]
        # N0[pktIdx] = np.power(data_set_ind, 2) * per
        # n = sqrt(N0[pktIdx]) * standard_normal(size=(104, 62, 1))
        # ====================CNN===========================
        if np.isnan(np.sum(n)) or np.isinf(np.sum(n)):
            data_add_AWGN[pktIdx] = s
        else:
            data_add_AWGN[pktIdx] = s + n
    return data_add_AWGN



# all the cleverhans
def eps2per(data_adv,data_ori, psrdB):
    pkt_num = data_ori.shape[0]
    # data_adv = data_adv.numpy()
    data_per_adv = data_adv - data_ori
    data_per_adv_ideal = np.empty(data_per_adv.shape, dtype="float")
    data_adv_psr = np.empty(data_per_adv.shape, dtype="float")
    psr = 10**(psrdB/10)
    data_per_ideal_power = compute_power(data_ori) * psr
    data_adv_power = compute_power(data_adv)
    for pktIdx in range(pkt_num):
        data_ori_ind = data_ori[pktIdx, :, :]
        data_adv_ind = data_adv[pktIdx, :, :]
        data_per_ind = data_per_adv[pktIdx, :, :]
        scale = np.mean(data_ori_ind**2) * psr
        # n = sqrt(scale) * abs(data_ori_ind - data_adv_ind)
        # data_per_adv_ideal[pktIdx] = sqrt(scale) * (data_per_ind / abs( data_per_ind ))
        p_x = np.mean(data_ori_ind**2)
        p_delta = np.mean(data_per_ind**2)
        data_per_adv_ideal[pktIdx] = np.sqrt(psr*p_x/(p_delta + 1e-30)) * data_per_ind
        # data_per_adv_ideal[pktIdx] = np.nan_to_num(data_per_adv_ideal[pktIdx])
        data_adv_psr[pktIdx] = data_ori_ind + data_per_adv_ideal[pktIdx]
        # scale = data_ideal_ind / data_adv_ind
        # scale = np.power(data_ideal_ind, 2) / np.power(data_adv_ind, 2)
        # N0 = np.empty(data_ori.shape)
        # N0[pktIdx] = np.power(data_ori_ind, 2) * psr
        # n = sqrt(N0[pktIdx]) * data_ori[pktIdx, :, :]
        # data_per_adv = abs(data_adv - data_ori)

        # scale = sqrt(psr) * data_ori[pktIdx, :, :] / data_per_adv[pktIdx, :, :]
        # scale = n / data_per_adv[pktIdx, :, :]

        # data_adv[pktIdx] = data_per_adv[pktIdx] * sqrt(scale)

        # n = sqrt(psr) * data_per[pktIdx, :, :]

        # if np.isnan(np.sum(n)) or np.isinf(np.sum(n)):
        #     data_adv_psr[pktIdx] = data_ori_ind
        # else:
        #     data_adv_psr[pktIdx] = data_ori_ind + n

    return data_adv_psr, data_per_adv_ideal

# for fgsm
def eps_per(data_adv,data_ori, psr):
    per = data_adv - data_ori
    pkt_num = data_ori.shape[0]
    per = per.numpy()
    data_adv_psr = np.empty(per.shape, dtype="float")
    for pktIdx in range(pkt_num):
        data_ori_ind = data_ori[pktIdx, :, :]
        data_per_ind = per[pktIdx, :, :]
        scale = np.power(data_ori_ind,2) * psr
        data_adv_psr[pktIdx] = sqrt(scale) * (data_per_ind / abs(data_per_ind))
        data_adv_psr[pktIdx] = np.nan_to_num(data_adv_psr[pktIdx])
    data_adv = data_adv_psr + data_ori
    # data_adv = np.nan_to_num(data_adv)
    return data_adv


def eps2per_uap(v, data_ori, psrdB):
    pkt_num = data_ori.shape[0]
    v_all = v.repeat(1000, axis=0)
    # data_adv = data_adv.numpy()
    data_per_adv_ideal = np.empty(data_ori.shape, dtype="float")
    psr = 10**(psrdB/10)
    data_per_ideal_power = compute_power(data_ori) * (1+psr)
    # data_adv_power = compute_power(v)

    for pktIdx in range(pkt_num):
        data_ori_ind = data_ori[pktIdx, :, :]
        data_adv_ind = data_adv[pktIdx, :, :]
        data_per_ind = data_per_adv[pktIdx, :, :]
        scale = np.mean(data_ori_ind**2) * psr
        # n = sqrt(scale) * abs(data_ori_ind - data_adv_ind)
        # data_per_adv_ideal[pktIdx] = sqrt(scale) * (data_per_ind / abs( data_per_ind ))
        p_x = np.mean(data_ori_ind**2)
        p_delta = np.mean(data_per_ind**2)
        data_per_adv_ideal[pktIdx] = np.sqrt(psr*p_x/p_delta) * data_per_ind
        data_per_adv_ideal[pktIdx] = np.nan_to_num(data_per_adv_ideal[pktIdx])
        data_adv_psr[pktIdx] = data_ori_ind + data_per_adv_ideal[pktIdx]
        # scale = data_ideal_ind / data_adv_ind
        # scale = np.power(data_ideal_ind, 2) / np.power(data_adv_ind, 2)
        # N0 = np.empty(data_ori.shape)
        # N0[pktIdx] = np.power(data_ori_ind, 2) * psr
        # n = sqrt(N0[pktIdx]) * data_ori[pktIdx, :, :]
        # data_per_adv = abs(data_adv - data_ori)

        # scale = sqrt(psr) * data_ori[pktIdx, :, :] / data_per_adv[pktIdx, :, :]
        # scale = n / data_per_adv[pktIdx, :, :]
        # data_adv[pktIdx] = data_per_adv[pktIdx] * sqrt(scale)
        # n = sqrt(psr) * data_per[pktIdx, :, :]


    return data_adv_psr

##############   acc does not change with psr
# def eps_per(data_adv,data_ori, psr):
#     pkt_num = data_ori.shape[0]
#     data_adv = data_adv.numpy()
#     data_adv_psr = np.empty(data_adv.shape, dtype="float")
#     for pktIdx in range(pkt_num):
#         data_ori_ind = data_ori[pktIdx, :, :]
#         data_adv_ind = data_adv[pktIdx, :, :]
#         scale = np.power(data_ori_ind,2) * (1+psr)
#         data_adv_psr[pktIdx] = sqrt((scale) / np.power(data_ori_ind,2)) * data_adv_ind
#     return data_adv_psr



def fourmension(data):
    num_sample = data.shape[0]
    num_row = 2
    num_column = data.shape[1]

    data_iq = np.zeros([num_sample, num_row, num_column, 1])
    for i in range(num_sample):
        data_iq[i,0,:,0] = np.real(data[i])
        data_iq[i,1,:,0] = np.imag(data[i])

    return data_iq


def covert_to_IQ(data):
    # result = data[:, 0, :, :];
    # result += tf.float(1.0j) * data[:, 1, :, :];

    result =tf.complex( data[:, 0, :, :] , data[:, 1, :, :] )

    result = np.squeeze(result)
    return result


def accuracy_target_score(y_target, y_pred, y_number):
    score = y_target == y_pred
    target_success_rate = score / y_number
    return target_success_rate

def calculate_success_rate(model_predict, target, n_trigger_added):
    return np.count_nonzero(model_predict == target) / n_trigger_added

def calculate_labeltotarget_rate(model_predict, target, n_trigger_added):
    # model_predict=model_predict[i:i+50]
    return np.count_nonzero(model_predict == target) / n_trigger_added

def compute_scale_factor( psr, v, ori_data ):
    v = np.array(v)
    n = v.shape[ 0 ]
    re_v = v.reshape( n, -1 )
    scale_factor = np.sqrt(
			psr * ori_data.var( ) * ((re_v.max( axis = 1 ) - re_v.min( axis = 1 )) ** 2) / re_v.var(
					axis = 1
					)
			)
    return scale_factor


def compute_power_adv( data_set_adv ):
    data_set_adv = np.array(data_set_adv,dtype=float)
    data_pow = 0
    if data_set_adv.ndim == 4 :
       data_set_adv = data_set_adv[:, :, :, 0]
       i = 0
       # data_pow = 0
       total_point_adv = data_set_adv.shape[1] + data_set_adv.shape[2]
       while i < data_set_adv.shape[0]:
           data_set_ind_adv = data_set_adv[i, :, :]
           data_pow = data_pow + np.power(data_set_ind_adv, 2).sum() / total_point_adv
           i = i + 1
       data_pow = data_pow / data_set_adv.shape[0]
    elif data_set_adv.ndim == 3 :
        i = 0
        # data_pow = 0
        total_point_adv = data_set_adv.shape[1] + data_set_adv.shape[2]
        while i < data_set_adv.shape[0]:
              data_set_ind = data_set_adv[i, :, :]
              data_pow = data_pow + np.power(data_set_ind, 2).sum() / total_point_adv
              i = i + 1
        data_pow = data_pow / data_set_adv.shape[0]
    return data_pow


# use every row's energy to calculate power
def compute_power( data_set ):
    if data_set.ndim == 4 :
       data_set = data_set[:, :, :, 0]
       i = 0
       data_set_power = 0
       total_point = data_set.shape[1] + data_set.shape[2]
       while i < data_set.shape[0]:
           data_set_ind = data_set[i, :, :]
           data_set_power = data_set_power + np.power(data_set_ind, 2).sum() / total_point
           i = i + 1
       data_set_power = data_set_power / data_set.shape[0]
    elif data_set.ndim == 3 :
        i = 0
        data_set_power = 0
        total_point = data_set.shape[1] + data_set.shape[2]
        while i < data_set.shape[0]:
              data_set_ind = data_set[i, :, :]
              data_set_power = data_set_power + np.power(data_set_ind, 2).sum() / total_point
              i = i + 1
        data_set_power = data_set_power / data_set.shape[0]
    return data_set_power


#using var to calculate power
def compute_power_var( data_set ):
    if data_set.ndim == 4 :
       data_set = data_set[:, :, :, 0]
       i = 0
       data_set_power = 0
       # total_point = data_set.shape[1] + data_set.shape[2]
       while i < data_set.shape[0]:
           data_set_ind = data_set[i, :, :]
           data_set_power = data_set_ind.var() + data_set_power
           i = i + 1
       data_set_power = data_set_power / data_set.shape[0]
    elif data_set.ndim == 3 :
        i = 0
        data_set_power = 0
        # total_point = data_set.shape[1] + data_set.shape[2]
        while i < data_set.shape[0]:
              data_set_ind = data_set[i, :, :]
              data_set_power = data_set_ind.var() + data_set_power
              i = i + 1
        data_set_power = data_set_power / data_set.shape[0]
    return data_set_power

# def adjust_uap_power(factor,data_test, v, clf):
#     v_1 = 1.2 * v
#     dataset_perturbed_v1 = data_test + v_1
#     dataset_perturbed_v1_power = compute_power(dataset_perturbed_v1)
#     data_set_power = compute_power(data_test):
#     psr_v1 = np.array(psr_v1)
#     psr_v1_dB = 10 * np.log10(psr_v1)
#     pred_prob_delay_v1 = clf.predict(dataset_perturbed_v1)
#     pred_label_delay_v1 = pred_prob_delay_v1.argmax(axis=-1)
#     acc_delay_v1 = accuracy_score(label_test, pred_label_delay_v1)
#
#     print('Overall accuracy_delay_v1 = %.4f ,psr = %.4f' % (acc_delay_v1, psr_v1_dB))
