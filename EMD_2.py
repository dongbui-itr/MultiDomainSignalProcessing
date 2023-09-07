#Paper: "https://www.hindawi.com/journals/wcmc/2020/8811962/"
'''[0] An Efficient ECG Denoising Method Based on Empirical Mode
Decomposition, Sample Entropy, and Improved Threshold Function'''
'''[0.1]An integrated EMD adaptive threshold denoising method for reduction of noise in
ECG'''

'''[1] Improved Electrode Motion Artefact Denoising in ECG Using Convolutional Neural Networks
and a Custom Loss Function'''

import time
import wfdb as wf
import os
from wfdb.processing import resample_sig
from utils.reprocessing import butter_bandpass_filter, smooth

from scipy.signal import welch, spectrogram
from scipy.fft import fftshift

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

fs = 250

file = '100'
ext_noise = ''

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
file_name = file +ext_noise

_mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
_file_name = file

start_smp = int(5*60*fs)
stop_smp = int(6*60*fs)
# stop_smp = start_smp + 30 * fs


def imf(signal, fs, tmp_sig=None, flag_sm=False,):
    upper_peaks, _ = find_peaks(signal, distance=0.01 * fs)
    lower_peaks, _ = find_peaks(-signal, distance=0.01 * fs)

    f1 = interp1d(upper_peaks / fs, signal[upper_peaks], kind='cubic', fill_value='extrapolate')
    f2 = interp1d(lower_peaks / fs, signal[lower_peaks], kind='cubic', fill_value='extrapolate')

    tAxis = np.arange(0, len(signal), 1) / fs

    y1 = f1(tAxis)
    y2 = f2(tAxis)
    y1[-5 :] = 0
    y2[-5 :] = 0

    if flag_sm:
        y1 = smooth(y1, int(fs/10))
        y2 = smooth(y2, int(fs/10))

    y = y1 - y2
    # _y = y.copy()
    y[np.abs(y) < 0.5] = 0

    # plt.plot(y, 'r', _y)
    # plt.show()

    if flag_sm :
        y = smooth(y, 20)

    # avg_envelope = ((y1 + y2) / 2 + y*2)/2
    avg_envelope = (y1 + y2) / 2

    res1 = avg_envelope
    imf1 = signal - avg_envelope
    # _signal, _ = resample_sig(signal, fs, fs*2)
    # imf1 = _signal - avg_envelope

    # plt.figure(figsize=(10, 8))
    #
    # plt.plot(tAxis, y1, label='upper envelope', color='g')
    # plt.plot(tAxis, y2, label='lower envelope', color='b')
    # plt.plot(tAxis, avg_envelope, label='average envelope', color='r')
    # plt.plot(tAxis, signal, label='signal', color='k')
    # plt.plot(tAxis, butter_bandpass_filter(signal, 0.5, 40, fs*2), label='signal_bp', color='y')
    # # plt.plot(tAxis, butter_bandpass_filter(avg_envelope, 0.5, 40, fs), label='average envelope', color='r')
    # plt.title('Visualizing envelopes')
    # # if not tmp_sig is None:
    # #     plt.plot(tAxis, tmp_sig, label='raw', color='m')
    # plt.xlabel('Time [s]')
    # plt.legend(loc='lower right')
    # plt.show()
    # plt.close()

    return res1, imf1, avg_envelope

def std_imf(imf):
    sigma = np.std(imf)
    _imf = imf.copy()
    imf[np.abs(imf) < (3 * sigma)] = 0
    # imf[imf < (sigma / 3)] = 0
    return imf


def main():
    data_raw = wf.rdsamp(os.path.join(_mitdb_path, _file_name))[0][:, 0]
    _x, _ = resample_sig(data_raw, 360, fs)

    data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
    x, _ = resample_sig(data_raw, 360, fs)

    x = x[start_smp: stop_smp]
    _x = _x[start_smp: stop_smp]

    list_imf = []
    list_imf_fft = []

    res1, imf1, avg1 = imf(x, fs, _x, flag_sm=False)
    imf1 = std_imf(imf1)
    list_imf.append(imf1)
    # res2, imf2, avg2 = imf(x - avg1, fs)
    res2, imf2, avg2 = imf(res1, fs, flag_sm=False)
    imf2 = std_imf(imf2)
    list_imf.append(imf2)
    # res3, imf3, avg3 = imf(x - avg2, fs)
    res3, imf3, avg3 = imf(res2, fs, flag_sm=False)
    imf3 = std_imf(imf3)
    list_imf.append(imf3)
    # res4, imf4, avg4 = imf(x-avg3, fs)
    res4, imf4, avg4 = imf(res3, fs)
    imf4 = std_imf(imf4)
    list_imf.append(imf4)

    len_win = 1024
    raw_fft = np.abs(fft(x, len_win))
    list_imf_fft.append(raw_fft)
    imf1_fft = np.abs(fft(imf1, len_win))
    list_imf_fft.append(imf1_fft)
    imf2_fft = np.abs(fft(imf2, len_win))
    list_imf_fft.append(imf2_fft)
    imf3_fft = np.abs(fft(imf3, len_win))
    list_imf_fft.append(imf3_fft)
    imf4_fft = np.abs(fft(imf4, len_win))
    list_imf_fft.append(imf4_fft)

    fft_freq = np.linspace(0, fs//2, len_win//2)

    tAxis = np.arange(0, len(x), 1) / fs

    plt.figure(num=1)
    sub_num = (len(list_imf) + 2) * 100 + 11
    ax1 = plt.subplot(sub_num)
    plt.plot(x)
    plt.title('Raw')
    for i in range(len(list_imf)):
        plt.subplot(sub_num + i + 1, sharex=ax1)
        plt.plot(list_imf[i])
        plt.title('IMF-{}'.format(i + 1))

    plt.subplot(sub_num + len(list_imf) + 1, sharex=ax1)
    plt.plot(res4)
    plt.title('Res4')


    # plt.figure(num=2)
    # sub_num = (len(list_imf_fft)) * 100 + 11
    # for i in range(len(list_imf_fft)):
    #     plt.subplot(sub_num + i)
    #     plt.plot(fft_freq, list_imf_fft[i][0:len_win//2])
    #     plt.title('FFT-IMF-{}'.format(i))

    plt.figure(num=3)
    plt.plot(butter_bandpass_filter(x, 0.5, 40, fs), 'b', label='raw_BP')

    # plt.plot(butter_bandpass_filter(imf2 + imf3 + imf4 + imf1 + res4, 0.5, 40, fs), 'g', label='EMD')
    # plt.plot(
    #     imf1 + \
    #           imf2 + \
    #     # butter_bandpass_filter(imf2, 0.5, 40, fs) + \
    #          # butter_bandpass_filter(imf1, 0.5, 20, fs) +
    #          imf3 + \
    #          #butter_bandpass_filter(imf3, 0.5, 40, fs) + \
    #          res3,
    #          # imf4 + \
    #          # butter_bandpass_filter(imf4, 0.5, 40, fs) + \
    #          # res4,
    #          # butter_bandpass_filter(res4, 0.5, 20, fs),
    #          'k', label='EMD')
    plt.plot(butter_bandpass_filter(_x, 0.5, 40, fs), alpha=0.6, color='m', label='origin_BP')
    plt.plot(x, 'k', label='raw', alpha=0.5)
    plt.plot(imf1 + imf2 + imf3 + imf4 + res4, color='r', label='EMD')
    plt.plot(smooth(butter_bandpass_filter(imf1 + imf2 + imf3 + imf4 + res4, 0.5, 40, fs)), color='g', label='EMD_bp')
    plt.legend()

    # plt.figure(num=4)
    # imf2_fft = np.abs(fft(butter_bandpass_filter(imf2, 1, 15, fs), len_win))
    # plt.plot(fft_freq, imf2_fft[0 :len_win // 2])

    plt.show()

main()