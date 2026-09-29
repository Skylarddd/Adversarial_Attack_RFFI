import numpy as np
# import h5py
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform, randn
import time
        
def doppler_spectrum(fd,Nfft):
    ''' Generating the Doppler Spectrum'''
    df=(2*fd)/(Nfft-1);
    f=np.arange(-fd, fd+df, df)
    S=1.5/(np.pi*fd*sqrt(1-(f/fd)**2))  #Jakes Doppler spectrum
    S[0]=2*S[1]-S[2]
    S[-1]=2*S[-2]-S[-3] 
    
    return S
    
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def exp_PDP(tau_d,fs):
    # tau_d: rms delay spread in second
    # fs: sampling frequency
    Ts = 1/fs
    A_dB = -30
    sigma_tau=tau_d 
    A=10**(A_dB/10)
    lmax = np.ceil(-tau_d*np.log(A)/Ts)
    
    # p0=1/sigma_tau
    p=np.arange(0,lmax+1) 
    path_delays = p*Ts
        
    p_total = sum((1/sigma_tau)*np.exp(-p*Ts/sigma_tau))
    p0 = 1/(sigma_tau*p_total);
    avg_pathgains = p0*np.exp(-p*Ts/sigma_tau);
    
    return avg_pathgains, path_delays


def gen_pathgains(avg_pathgains, v_ms, center_freq, fs, num_samples):
    
    num_taps = len(avg_pathgains)
    path_gains = np.zeros([num_samples, num_taps], dtype=complex)
    
    for i in range(num_taps):
        path_gains[:,i] = avg_pathgains[i] * rayleigh_ch_SOS(v_ms, center_freq, fs, num_samples)
        
    return path_gains


def channel_filter(ch_in, path_gains):    
    num_tap = path_gains.shape[1]
    
    ch_out = np.zeros(len(ch_in) + num_tap, dtype=complex) 
    for i in range(len(ch_in)):     # loop for len(ch_in) times
        ch_out[i:i+num_tap] += ch_in[i]*path_gains[i]
    # print('done')
    return ch_out



def awgn(data, snr_range):
    
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        data[pktIdx] = s + n

    return data 


def channel_aug(data,center_freq,fs, speed_range, rms_delay_range):
    
    pkt_num = data.shape[0]
    sig_len = data.shape[1]
    
    v_ms = uniform(speed_range[0],speed_range[-1],pkt_num)  # random moving speed
    tau_d = uniform(rms_delay_range[0],rms_delay_range[-1],pkt_num)*1e-9 # random rms delay spread   
    
    data_augmented = np.zeros([pkt_num,sig_len], dtype=complex)

    pad_len = round(0.01*sig_len)
    for pkt_idx in range(pkt_num):     
        avg_pathgains, path_delays = exp_PDP(tau_d[pkt_idx],fs)
        path_gains = gen_pathgains(avg_pathgains, 
                                   v_ms[pkt_idx], 
                                   center_freq, 
                                   fs, 
                                   sig_len + pad_len)
        
        ch_in = np.pad(data[pkt_idx],(0,pad_len))
        ch_out = channel_filter(ch_in,path_gains) # ch_out = np.ones(sig_len,dtype = complex)
        ch_out = ch_out[0:sig_len]
        
        data_augmented[pkt_idx] = ch_out

    return data_augmented
    