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
from utils.reprocessing import butter_bandpass_filter, smooth, bwr

from scipy.signal import welch, spectrogram
from scipy.fft import fftshift

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from PyEMD import EEMD, CEEMDAN, EMD

fs = 250

file = '119'
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
    for i in range((E_IMFs.shape[0])):
        if i < 4: # or i > 5:
            E_IMFs[i] = std_imf(E_IMFs[i], 2)
        else:
            E_IMFs[i] = std_imf(E_IMFs[i], 1.5)
        #     break

    E_IMFs[-1] = bwr(E_IMFs[-1], fs)

    plt.figure(num=1)
    sub_num = ((E_IMFs.shape[0]//denominator + 1) * 100 + 11)
    ax1 = plt.subplot(sub_num)
    plt.plot(x)
    plt.title('Raw')
    for i in range((E_IMFs.shape[0]//denominator)):
        # print(i)
        plt.subplot(sub_num + i + 1, sharex=ax1)
        plt.plot(E_IMFs[i])
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

    plt.plot(butter_bandpass_filter(reconstruction, 0.5, 40, fs), color='r', label='re')
    plt.plot(butter_bandpass_filter(x, 0.5, 40, fs), color='g', label='raw')

    plt.plot(smooth(bwr(reconstruction, fs)), color='k', label='re')
    # plt.plot(bwr(x, fs), color='g', label='raw')

    plt.show()


    imfNo = E_IMFs.shape[0]



main()