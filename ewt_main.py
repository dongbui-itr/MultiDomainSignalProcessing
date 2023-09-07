import os
import wfdb as wf
import numpy as np

from utils.reprocessing import butter_bandpass_filter, bwr

import scipy as sc
import scipy.signal as signal
from wfdb.processing import resample_sig
from glob import glob
from matplotlib import pyplot as plt
import librosa
import ewtpy


fs = 128
file = '119'

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/nstdb/'
file_name = file + 'e06'

_mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
# file_name = '100'

data_org = wf.rdsamp(os.path.join(_mitdb_path, file_name[:-3]))[0][:, 0]
data_org, _ = resample_sig(data_org, 360, fs)
data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
data, _ = resample_sig(data_raw, 360, fs)

start_smp = int(4.5*60*fs)
stop_smp = int(5.5*60*fs)

data = data[start_smp:stop_smp]

ewt,  mfb, boundaries = ewtpy.EWT1D(data, N=9)

num_subplot = ewt.shape[-1]
num = num_subplot * 100 + 10 + 1
for isub in range(num_subplot):
    plt.subplot(num + isub)
    plt.plot(ewt[:, isub])

plt.show()

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(data)
# plt.title('Original signal')
# plt.xlabel('time (s)')
# plt.subplot(2,1,2)
# plt.plot(ewt)
# plt.plot(data_org, alpha=0.6)
# plt.title('Decomposed signals')
# plt.xlabel('time (s)')
# plt.legend(['Mode'])
# plt.tight_layout()
#
# # Plot raw
# plt.figure(figsize=(16,2))
# plt.plot(data, 'r')
# plt.xlabel("Time [s]")
# plt.ylabel("raw data")
# plt.show()