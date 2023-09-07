import numpy as np
import matplotlib.pyplot as plt
from constantQ.timeseries import TimeSeries
import time
import wfdb as wf
import os
from wfdb.processing import resample_sig
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
file_name = '119e06'

_mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
_file_name = '119'


def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


data_raw = wf.rdsamp(os.path.join(_mitdb_path, _file_name))[0][:, 0]
_x, _ = resample_sig(data_raw, 360, f1)

data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
x, _ = resample_sig(data_raw, 360, f1)


# Constant Q Transform - not properly labeled
# series = TimeSeries(x[int(4.75*60*f1):int(5.75*60*f1)], dt=1/f1, unit='m', name='test', t0=0)     #np.array --> gwpy.timeseries
series = TimeSeries(normalize(x[int(4.75*60*f1):int(5.75*60*f1)]), dt=1/f1, unit='m', name='test', t0=0)     #np.array --> gwpy.timeseries
hdata = series
dstTime = time.time()
sq = hdata.q_transform(search=None)

n_fft = 512
hop_length = n_fft // 2 #int(fs * 5)
win_length = hop_length//5

# cqt_cal = librosa.cqt(x[int(4.75*60*f1):int(5.75*60*f1)], hop_length=56, n_bins=100, sr=f1)
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


print(len(sq[0]))       # freq N
print(len(sq))          # time N

plt.figure(1)
plt.subplot(211)
plt.plot(_x[int(4.75*60*f1):int(5.25*60*f1)])
plt.subplot(212)
plt.plot(x[int(4.75*60*f1):int(5.25*60*f1)])

plt.figure(2)
current = time.time()
print('DST Time: '+str(current - dstTime))
plt.imshow(sq.T, origin='lower')
#plt.pcolor(sq.T)
# plt.colorbar()
plt.show()

# # Discrete Time Fourier Transform - not properly labeled
# from scipy import signal as scisignal
# dtftTime = time.time()
# freq, ts, Sxx = scisignal.spectrogram(x)
# print('DTFT Time: '+str(time.time() - dtftTime))
#
# plt.pcolor(ts, freq, Sxx, shading='auto')
# plt.colorbar()
# plt.show()