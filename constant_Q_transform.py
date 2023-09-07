import numpy as np
import matplotlib.pyplot as plt
from constantQ.timeseries import TimeSeries
import time
import wfdb as wf
import os
from wfdb.processing import resample_sig
from  utils.reprocessing import butter_bandpass_filter

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

start_smp = int(5.5*60*f1)
# stop_smp = int(5.5*60*f1)
stop_smp = start_smp + 10 * f1


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

series = TimeSeries(_signal, dt=1/f1, unit='m', name='test', t0=0)     #np.array --> gwpy.timeseries
# series = TimeSeries(normalize(x[start_smp: stop_smp]), dt=1/f1, unit='m', name='test', t0=0)     #np.array --> gwpy.timeseries
hdata = series
dstTime = time.time()
sq = hdata.q_transform(search=None, fftlength=fftlength/f1, overlap=overlap/f1)
current = time.time()
_sq = hdata.q_transform(search=None, fftlength=fftlength/f1, overlap=overlap/f1)
sq = np.asarray(sq)

print('Max spectrogram: ', np.max(sq))
print('Max spectrogram < 50Hz: ', np.max(sq[:, :50]))
print('Max spectrogram > 50Hz: ', np.max(sq[:, 50:]))

# plt.figure(2)
#
# print('DST Time: '+str(current - dstTime))
# # plt.imshow(sq.T, origin='lower')
# plt.imshow(np.array(sq).T)
# #plt.pcolor(sq.T)
# # plt.colorbar()
# plt.show()

# sq_50 = sq[:, :75]
# sq_50_norm = (sq_50 - np.min(sq_50)) / (np.max(sq_50) - np.min(sq_50))
# sq[:, :75] = sq_50_norm
#
# sq_o50 = sq[:, 75:]
# sq_o50_norm = (sq_o50 - np.min(sq_o50)) / (np.max(sq_o50) - np.min(sq_o50))
# sq[:, 75:] = sq_o50_norm

# n_fft = 256
# hop_length = n_fft // 2 #int(fs * 5)
# win_length = hop_length//5

# cqt_cal = librosa.cqt(x[start_smp: stop_smp], hop_length=256, n_bins=100, sr=f1, fmin=0.001)
#
# from main_4_noise_detection import plot_spectrogram
# plot_spectrogram(librosa.amplitude_to_db(cqt_cal), '', x[start_smp: stop_smp])
#
#
# f_range = np.arange(0.001, f1//2, f1//2/100)
# indx = (f_range <= 0.5) | (f_range >= 40)
# # cqt_cal[indx] = 0
#
# x_restore = librosa.icqt(cqt_cal, hop_length=256, sr=f1, fmin=0.001)


# cqt_cal = librosa.hybrid_cqt(x[int(4.75*60*f1):int(5.75*60*f1)], sr=f1, fmin=0.01, hop_length=48, n_bins=1000 )
# f_range = np.arange(0.001, f1//2, f1//2/1000)
# indx = (f_range <= 0.5) | (f_range >= 40)

# cqt_cal = librosa.cqt(x[int(4.75*60*f1):int(5.75*60*f1)], sr=f1, fmin=0.001, hop_length=48, n_bins=100)
# _cqt_cal = librosa.cqt(x[int(4.75*60*f1):int(5.75*60*f1)], sr=f1, fmin=0.001, hop_length=48, n_bins=100)
# f_range = np.arange(0.001, f1//2, f1//2/100)
# indx = (f_range <= 0.5) | (f_range >= 40)
#
#
# cqt_cal[indx] = 0
#
# x_restore = librosa.icqt(cqt_cal, sr=f1, fmin=0.001, hop_length=48)
#
# plt.plot(x[int(4.75*60*f1):int(5.75*60*f1)], 'r', x_restore, 'b')
# plt.show()

# cqt_cal_db = librosa.core.amplitude_to_db(cqt_cal, ref=1.0, amin=1e-20, top_db=80.0)
# from main_2 import plot_spectrogram
# plot_spectrogram(cqt_cal_db, 'cqt', x[int(4.75*60*f1):int(5.75*60*f1)])


# print(len(sq[0]))       # freq N
# print(len(sq))          # time N

# plt.figure(1)
# plt.subplot(211)
# plt.plot(_x[start_smp: stop_smp])
# plt.plot(x_restore)
# plt.subplot(212)
# plt.plot(x[start_smp: stop_smp])
# plt.figure(2)
# plt.plot(np.abs(cqt_cal).T)

# sq_repeat_1000 = np.repeat(sq, repeats=1000, axis=0)
# sq_repeat_1000_o50 = np.repeat(sq[:, 50:], repeats=15, axis=0)
sq_repeat_1000_o50 = np.repeat(sq[:, :50], repeats=15, axis=0)
# sq_repeat_1000_o50 = (sq_repeat_1000_o50 - np.min(sq_repeat_1000_o50)) / (np.max(sq_repeat_1000_o50) - np.min(sq_repeat_1000_o50))
# sq_repeat_1000_o50 = (sq_repeat_1000_o50 - np.min(sq_repeat_1000_o50)) / (np.max(sq_repeat_1000_o50) - np.min(sq_repeat_1000_o50))

# plt.imshow(sq[:, 50:].T)
# plt.show()

tmp = np.zeros(len(x[start_smp:stop_smp]))
# for fs in range(50, sq.shape[-1], 1):
for fs in range(0, 30, 1):
    t = np.arange(0, 60, 1/f1)
    # sine_wave = (1/(fs - 50 + 1)) * np.sin(2 * np.pi * fs * t)
    sine_wave = (1/(fs + 1)) * np.sin(2 * np.pi * fs * t)
    # plt.plot(sine_wave)
    # plt.show()
    # plt.plot(np.multiply(np.log10(sq_repeat_1000_o50)[:, fs - 50], sine_wave))
    # plt.show()
    # x[start_smp:stop_smp] -= np.multiply(np.log(sq_repeat_1000_o50)[:, fs - 50], sine_wave)
    # plt.plot(np.multiply(sq_repeat_1000_o50[:, fs - 50], sine_wave))
    # plt.show()
    # tmp += np.multiply(sq_repeat_1000_o50[:, fs - 50], sine_wave)
    tmp += np.multiply(sq_repeat_1000_o50[:, fs], sine_wave)

    a=10


plt.plot(x[start_smp:stop_smp] + tmp, alpha=0.5)
# plt.plot(x[start_smp:stop_smp], alpha=0.5)
# plt.plot(tmp, alpha=0.5)

plt.plot(_x[start_smp:stop_smp], alpha=0.5)
# plt.plot(tmp)
plt.show()

# print('Max spectrogram: ', np.max(sq))
# print('Max spectrogram < 50Hz: ', np.max(sq[:, :50]))
# print('Max spectrogram > 50Hz: ', np.max(sq[:, 50:]))
#
# plt.figure(2)
#
# print('DST Time: '+str(current - dstTime))
# # plt.imshow(sq.T, origin='lower')
# plt.imshow(np.array(sq).T)
# #plt.pcolor(sq.T)
# # plt.colorbar()
# plt.show()

# # Discrete Time Fourier Transform - not properly labeled
# from scipy import signal as scisignal
# dtftTime = time.time()
# freq, ts, Sxx = scisignal.spectrogram(_x[start_smp: stop_smp], fs=f1, nfft=512, nperseg=128)
# # freq, Sxx = scisignal.welch(x[start_smp: stop_smp], fs=f1, nfft=1024, nperseg=128)
# print('DTFT Time: '+str(time.time() - dtftTime))
# plt.figure(3)
# plt.pcolor(ts, freq, Sxx, shading='auto')
# plt.colorbar()
#
# # plt.plot(freq, Sxx)
# plt.show()