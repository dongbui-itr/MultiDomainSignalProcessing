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
from utils.reprocessing import butter_bandpass_filter, smooth, bwr, butter_highpass_filter

from scipy.signal import welch, spectrogram
from scipy.fft import fftshift

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from PyEMD import EEMD, CEEMDAN, EMD

fs = 250

file = '118'
ext_noise = 'e06'

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/nstdb/'
file_name = file +ext_noise

_mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
_file_name = file

start_smp = int(4.5 * 60 * fs)
stop_smp = int(5.5 * 60 * fs)
# stop_smp = start_smp + 5 * fs


def std_imf(imf, factor=3):
    sigma = np.std(imf)
    _imf = imf.copy()
    imf[np.abs(imf) < (factor * sigma)] = 0
    # imf[imf < (sigma / 3)] = 0
    return imf


def rm_imf(imf, factor=3):
    imf_bk = imf.copy()
    while True:
        sigma = np.std(imf)
        tmp_mean_imf = np.mean(imf)
        tmp_imf = imf - tmp_mean_imf
        indx = np.flatnonzero(tmp_imf < (factor * sigma))
        if len(indx) > 0:
            imf_bk[indx] = 0
            imf = np.delete(imf, indx)
        else:
            break

    return imf_bk


def main():
    data_raw = wf.rdsamp(os.path.join(_mitdb_path, _file_name))[0][:, 0]
    _x, _ = resample_sig(data_raw, 360, fs)

    data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
    x, _ = resample_sig(data_raw, 360, fs)

    x = x[start_smp: stop_smp]
    _x = _x[start_smp: stop_smp]

    # EEMD options
    max_imf = -1

    # Prepare and run EEMD
    # ceemdan = CEEMDAN(trials=7)
    eemd = EEMD(trials=7)
    # ceemdan.noise_seed(12345)

    tAxis = np.arange(0, len(x), 1) / fs

    # E_IMFs = ceemdan.ceemdan(x, tAxis, max_imf)
    E_IMFs = eemd.eemd(x, tAxis, max_imf)

    # plt.figure(num=1)
    # sub_num = ((E_IMFs.shape[0]//2 + 1) * 100 + 11)
    # ax1 = plt.subplot(sub_num)
    # plt.plot(x)
    # plt.title('Raw')
    # for i in range((E_IMFs.shape[0]//2)):
    #     # print(i)
    #     plt.subplot(sub_num + i + 1, sharex=ax1)
    #     plt.plot(E_IMFs[i])
    #     plt.title('IMF-{}'.format(i + 1))
    #
    # plt.figure(num=2)
    # sub_num = ((E_IMFs.shape[0] - E_IMFs.shape[0] // 2 + 1) * 100 + 11)
    # ax1 = plt.subplot(sub_num)
    # plt.plot(x)
    # plt.title('Raw')
    # for i in range((E_IMFs.shape[0] - E_IMFs.shape[0] // 2)) :
    #     print(i)
    #     plt.subplot(sub_num + i + 1, sharex=ax1)
    #     plt.plot(E_IMFs[i + (E_IMFs.shape[0] // 2)])
    #     plt.title('IMF-{}'.format(i + 1 + (E_IMFs.shape[0] // 2)))
    #
    # plt.show()

    denominator = 2
    _E_IMFs = E_IMFs.copy()
    sigma = []
    sigma_hat = []
    for i in range((E_IMFs.shape[0])):
        #1) Calculate the standard deviation of an IMF
        tmp_sigma = np.std(E_IMFs[i])
        sigma.append(tmp_sigma)
        #2) Estimate the standard deviation of noise in the corresponding IMF
        tmp_mean = np.mean(E_IMFs[i])
        sigma_hat.append(np.median(np.abs(E_IMFs[i] - tmp_mean)) / 0.6745)

    sigma = np.asarray(sigma)
    sigma_hat = np.asarray(sigma_hat)
    #3) Estimate the boundary of noise dominating IMF
    kb = (np.argmax(np.divide(sigma, sigma_hat)) + np.argmax(np.subtract(sigma, sigma_hat))) // 2

    for i in range((E_IMFs.shape[0])) :
        if i <= kb:
            E_IMFs[i] = std_imf(E_IMFs[i], 3)
        else:
            # E_IMFs[i] = bwr(E_IMFs[i], fs)
            E_IMFs[i] = butter_highpass_filter(E_IMFs[i], 1, fs)

    # E_IMFs[-1] = bwr(E_IMFs[-1], fs)

    plt.figure(num=1)
    sub_num = ((E_IMFs.shape[0]//denominator + 1) * 100 + 11)
    ax1 = plt.subplot(sub_num)
    plt.plot(x)
    plt.title('Raw')
    for i in range((E_IMFs.shape[0]//denominator)):
        # print(i)
        plt.subplot(sub_num + i + 1, sharex=ax1)
        plt.plot(E_IMFs[i])
        plt.plot(x)
        plt.title('IMF-{}'.format(i + 1))

    plt.figure(num=2)
    sub_num = ((E_IMFs.shape[0] - E_IMFs.shape[0] // denominator + 1) * 100 + 11)
    ax1 = plt.subplot(sub_num)
    plt.plot(x)
    plt.title('Raw')
    for i in range((E_IMFs.shape[0] - E_IMFs.shape[0] // denominator)) :
        print(i)
        plt.subplot(sub_num + i + 1, sharex=ax1)
        plt.plot(E_IMFs[i + (E_IMFs.shape[0] // denominator)])
        plt.title('IMF-{}'.format(i + 1 + (E_IMFs.shape[0] // denominator)))

    plt.figure(num=3)
    reconstruction = np.zeros_like(x)
    for i in range(E_IMFs.shape[0] - 1) :
        reconstruction += E_IMFs[i]

    reconstruction_bp = butter_bandpass_filter(reconstruction, 0.5, 40, fs)
    x_bp = butter_bandpass_filter(x, 0.5, 40, fs)
    _x_bp = butter_bandpass_filter(_x, 0.5, 40, fs)

    # plt.plot(reconstruction_bp, color='r', label='re')
    plt.plot(x_bp, color='g', label='raw_NSTDB_bp')
    plt.plot(_x_bp, color='m', label='raw_MITDB_bp')
    plt.plot(np.multiply(x_bp, np.abs(reconstruction_bp)), color='b', label='New')
    # plt.plot(np.add(x_bp, np.abs(reconstruction_bp)), color='b', label='ext')
    plt.legend()

    # plt.plot(smooth(bwr(reconstruction, fs)), color='k', label='re')
    # plt.plot(bwr(x, fs), color='g', label='raw')

    plt.show()


    imfNo = E_IMFs.shape[0]



main()