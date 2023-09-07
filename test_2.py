import os
import numpy as np
import matplotlib
from  matplotlib import pyplot as plt
import librosa
from ipywidgets import interact, fixed, FloatSlider
import IPython.display as ipd

from wfdb.processing import resample_sig
import wfdb as wf
import os

from utils.reprocessing import butter_bandpass_filter, bwr

# %matplotlib inline

Fs = 128

duration = 10
omega1 = 1
omega2 = 5
N = int(duration * Fs)
t = np.arange(N) / Fs
t1 = t[:N//2]
t2 = t[N//2:]

x1 = 1.0 * np.sin(2 * np.pi * omega1 * t1)
x2 = 0.7 * np.sin(2 * np.pi * omega2 * t2)
x = np.concatenate((x1, x2))

# plt.figure(figsize=(8, 2))
# plt.subplot(1, 2, 1)
# plt.plot(t, x, c='k')
# plt.xlim([min(t), max(t)])
# plt.xlabel('Time (seconds)')
#
# plt.subplot(1, 2, 2)
# X = np.abs(np.fft.fft(x)) / Fs
# freq = np.fft.fftfreq(N, d=1/Fs)
# X = X[:N//2]
# freq = freq[:N//2]
# plt.plot(freq, X, c='k')
# plt.xlim([0, 7])
# plt.ylim([0, 3])
# plt.xlabel('Frequency (Hz)')
# plt.tight_layout()

fs = 250

file = '100'
mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
file_name = file# + 'e24'

data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
x, _ = resample_sig(data_raw, 360, fs)

x = butter_bandpass_filter(x, 0.5, 30, fs)


samp_start = int(4.5 * 60 * fs)
samp_end = int(5.5 * 60 * fs)

x = x[samp_start: samp_end]

t = np.arange(len(x)) / fs
# duration = len(x)/fs

def windowed_ft(t, x, Fs, w_pos_sec, w_len) :
    N = len(x)
    w_pos = N #int(Fs * w_pos_sec)
    w_padded = np.zeros(N) + 1
    w_padded[w_pos :w_pos + w_len] = 1
    x = x * w_padded
    plt.figure(figsize=(8, 2))



    X = np.abs(np.fft.fft(x)) / Fs
    freq = np.fft.fftfreq(N, d=1 / Fs)


    X[X < 0.3] = 0
    x_inver = np.abs(np.fft.ifft(X))

    X = X[:N // 2]
    freq = freq[:N // 2]

    plt.subplot(1, 2, 1)
    plt.plot(t, x, c='k')
    plt.plot(t, x_inver, c='g')
    plt.plot(t, w_padded, c='r')
    plt.xlim([min(t), max(t)])
    # plt.ylim([-1.1, 1.1])
    plt.xlabel('Time (seconds)')

    plt.subplot(1, 2, 2)

    plt.plot(freq[2:], X[2:], c='k')
    # plt.xlim([0, 7])
    # plt.ylim([0, 3])
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

Fs = fs
w_len = 4 * Fs

windowed_ft(t, x, Fs, w_pos_sec=1, w_len=w_len)
windowed_ft(t, x, Fs, w_pos_sec=3, w_len=w_len)
windowed_ft(t, x, Fs, w_pos_sec=5, w_len=w_len)

print('Interactive interface for experimenting with different window shifts:')
interact(windowed_ft,
         w_pos_sec=FloatSlider(min=0, max=duration - (w_len / Fs), step=0.1,
                               continuous_update=False, value=1.7, description='Position'),
         t=fixed(t), x=fixed(x), Fs=fixed(Fs), w_len=fixed(w_len));