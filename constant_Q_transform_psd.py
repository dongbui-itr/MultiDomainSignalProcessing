import numpy as np
import matplotlib.pyplot as plt
from constantQ.timeseries import TimeSeries
import time
import wfdb as wf
import os
from wfdb.processing import resample_sig
from  utils.reprocessing import butter_bandpass_filter

from scipy.signal import welch, spectrogram
from scipy.fft import fftshift

import librosa

# Generate np.array chirp signal
# dt = 0.001
# t = np.arange(0,3,dt)
# f0 = 50
f1 = 250
# t1 = 2
# x = np.cos(2*np.pi*t*(f0 + (f1 - f0)*np.power(t, 2)/(3*t1**2)))
# fs = 1/dt

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/nstdb/'
file_name = '118e24'

_mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
_file_name = '118'

start_smp = int(4.5*60*f1)
stop_smp = int(5.5*60*f1)
# stop_smp = start_smp + 10 * f1


def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


data_raw = wf.rdsamp(os.path.join(_mitdb_path, _file_name))[0][:, 0]
_x, _ = resample_sig(data_raw, 360, f1)

data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
x, _ = resample_sig(data_raw, 360, f1)

# x = butter_bandpass_filter(x, 0.5, 40, f1)

# Constant Q Transform - not properly labeled
signal = x[start_smp:stop_smp]

fftlength = 512
overlap = 452

len_padding = len(signal) % (fftlength - overlap)
if len_padding != 0:
    len_padding = fftlength - len_padding

_signal = np.concatenate((signal, np.zeros(len_padding)))


f, pxx = welch(_signal, fs=f1, window='hann', nfft=fftlength, nperseg=overlap)
# f2, t2, sxx = spectrogram(_signal, fs=f1, window='hann', nfft=fftlength) # , nperseg=overlap)

# plt.pcolormesh(t2, fftshift(f2), fftshift(sxx, axes=0), shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


a=10



