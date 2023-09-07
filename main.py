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
fs = 250
file = '119'

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/nstdb/'
file_name = file + 'e06'

_mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
# file_name = '100'


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x):
    return librosa.core.db_to_amplitude(x)

def main():
    f_low = 0.5
    f_high = 30

    data_org = wf.rdsamp(os.path.join(_mitdb_path, file_name[:-3]))[0][:, 0]
    data_org, _ = resample_sig(data_org, 360, fs)
    data_org_bp = butter_bandpass_filter(data_org, f_low, f_high, fs)

    data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
    data, _ = resample_sig(data_raw, 360, fs)

    samp_start = int(4.5 * 60 * fs)
    samp_end = int(5.5 * 60 * fs)
    data = data[samp_start:samp_end]

    data_bp = butter_bandpass_filter(data, f_low, f_high, fs)
    len_data = len(data)
    ##### data ==> STFT ==> THR ==> restore ==> FFT ==> THR ==> restore ####
    f_stft, t_stft, data_stft = signal.stft(data, fs, nfft=2048, nperseg=256, noverlap=128)
    # indx = f_stft <= f_low
    # data_stft[indx] = 0
    # indx = f_stft >= f_high + 10
    # data_stft[indx] = 0
    # _, data_restore_stft = signal.istft(data_stft, fs)
    #
    # f_stft, t_stft, _data_stft = signal.stft(data, fs)
    # data_stft_db = _amp_to_db(data_stft ** 2)
    # data_stft_db_in_flow_fhigh = data_stft_db[np.flatnonzero((f_stft <= f_high) & (f_stft >= f_low))]
    # data_stft_db_lower_flow = data_stft_db[np.flatnonzero(f_stft < f_low)]
    # data_stft_db_higher_fhigh = data_stft_db[np.flatnonzero(f_stft > f_high)]

    # data_stft_db_in_flow_fhigh_flat = np.unique(np.sort(data_stft_db_in_flow_fhigh.flatten()), return_index=True, return_counts=True)
    # data_stft_db_out_flow_fhigh_flat = np.unique(np.sort(data_stft_db_out_flow_fhigh.flatten()), return_index=True, return_counts=True)

    # data_stft_db_lower_flow[data_stft_db_lower_flow > np.max(data_stft_db_in_flow_fhigh)] = np.max(data_stft_db_in_flow_fhigh)
    # data_stft_db_higher_fhigh[data_stft_db_higher_fhigh > np.max(data_stft_db_in_flow_fhigh)] = np.max(data_stft_db_in_flow_fhigh)
    # data_stft_db_restore = np.concatenate((data_stft_db_lower_flow, data_stft_db_in_flow_fhigh, data_stft_db_higher_fhigh))
    #
    # data_stft_restore = _db_to_amp(data_stft_db_restore)

    data_stft_mod = np.abs(data_stft)

    plt.figure(1)
    plt.pcolormesh(t_stft, f_stft, data_stft, shading='gouraud')
    plt.ylim([f_stft[1], f_stft[-1]])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('log')

    plt.figure(2)
    plt.plot(data)
    # plt.xticks(np.arange(0, len(data), 500)/fs)
    plt.show()


    # data_stft[f_stft <= f_low] = 0
    # data_stft[f_stft >= f_high] = 0
    # _, data_restore_stft = signal.istft(data_stft, fs, nfft=1024)
    # # _, data_restore_stft_rm = signal.istft(data_stft_restore, fs)
    # a=10
    # _data_restore_stft_rm = bwr(data, fs) - bwr(-data_restore_stft_rm, fs)
    # _data_restore_stft_rm = data - data_restore_stft[:len(data)]
    # _data_restore_stft_rm = data - data_restore_stft_rm[:len(data)]

    # ax = plt.subplot(211)
    # ax.matshow(
    #     data_stft_db,
    #     origin="lower",
    #     aspect="auto",
    #     cmap=plt.cm.seismic,
    #     vmin=-1 * np.max(np.abs(data_stft_db)),
    #     vmax=np.max(np.abs(data_stft_db)),
    # )

    plt.subplot(212)
    plt.plot(data, 'r', label='raw')
    plt.plot(data_restore_stft, 'b', label='stft')
    # plt.plot(data_restore_stft_rm, 'g', label='stft_rm')
    # plt.plot(_data_restore_stft_rm, 'r', label='_stft_rm', alpha=0.6)
    plt.plot(data_org_bp[samp_start :samp_end], 'k', label='raw', alpha=0.6)
    plt.legend()
    plt.show()

    # ##### data ==> FFT ==> THR ==> restore ==> STFT ==> THR ==> restore ####
    # data_fft = sc.fft.rfft(data)
    # _data_fft = sc.fft.rfft(data, fs)
    # f_fft = sc.fft.rfftfreq(len_data, fs/len_data)
    # data_fft[f_fft <= (1e1)] = 0
    # data_fft[f_fft > 1e5] = 0
    # data_restore_fft = sc.fft.irfft(data_fft)
    #
    # data_restore_fft = np.concatenate((data_restore_fft, np.zeros(len(data) - len(data_restore_fft))))
    # data_combile_stft_fft = np.add(data_restore_fft, data_restore_stft[:len(data_restore_fft)])/2
    #
    # plt.plot(data, 'r', label='raw')
    # plt.plot(data_restore_stft, 'b', label='stft')
    # plt.plot(data_restore_fft, 'y', label='fft')
    # plt.plot(data_combile_stft_fft, 'm', label='stft_fft')
    # plt.plot(data_bp, 'g', label='bpf')
    # # plt.plot(data_org_bp, 'k', label='bp_org', alpha=0.5)
    #
    # plt.legend()
    # plt.show()
    # plt.interactive(False)

    a1=0


if __name__ == '__main__':
    main()