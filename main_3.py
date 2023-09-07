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
fs = 128
file = '119'

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/nstdb/'
file_name = file + 'e_6'

_mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
# file_name = '100'


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x):
    return librosa.core.db_to_amplitude(x, ref=1.0)

# def main():
#     f_low = 0.5
#     f_high = 30
#
#     data_org = wf.rdsamp(os.path.join(_mitdb_path, file_name[:-3]))[0][:, 0]
#     data_org, _ = resample_sig(data_org, 360, fs)
#     data_org_bp = butter_bandpass_filter(data_org, f_low, f_high, fs)
#
#     data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
#     data_nst, _ = resample_sig(data_raw, 360, fs)
#
#     data_bp = butter_bandpass_filter(data_nst, f_low, f_high, fs)
#
#     samp_start = int(4.5 * 60 * fs)
#     samp_end = int(5.5 * 60 * fs)
#     data = data_nst[samp_start:samp_end]
#
#     len_data = len(data)
#     ##### data ==> STFT ==> THR ==> restore ==> FFT ==> THR ==> restore ####
#     f_stft, t_stft, data_stft = signal.stft(data, fs, nperseg=16, nfft=64, noverlap=None)
#     # plt.pcolormesh(t_stft, f_stft, np.abs(data_stft))
#     # plt.show()
#
#
#     # f_stft, t_stft, _data_stft = signal.stft(data, fs)
#     indx_1 = f_stft <= f_low
#     data_stft[f_stft <= f_low] = 0
#     indx_2 = f_stft >= f_high
#     data_stft[f_stft >= f_high] = 0
#     _, data_restore_stft = signal.istft(data_stft, fs, nperseg=16, nfft=64, noverlap=None)
#
#     data_stft_fft = sc.fft.rfft(data_restore_stft[: len(data)])
#     f_stft_fft = sc.fft.rfftfreq(len_data, fs/len_data)
#     data_stft_fft[f_stft_fft <= f_low * 2] = 0
#     data_stft_fft[f_stft_fft > f_high / 2] = 0
#     data_restore_stft_fft = sc.fft.irfft(data_stft_fft)
#
#
#     ##### data ==> FFT ==> THR ==> restore ==> STFT ==> THR ==> restore ####
#     data_fft = sc.fft.rfft(data)
#     _data_fft = sc.fft.rfft(data, fs)
#     f_fft = sc.fft.rfftfreq(len_data, fs/len_data)
#     data_fft[f_fft <= f_low] = 0
#     data_fft[f_fft > f_high] = 0
#     data_restore_fft = sc.fft.irfft(data_fft)
#
#     f_fft_stft, t_fft_stft, data_fft_stft = signal.stft(data_restore_fft, fs, nperseg=1024, nfft=2048, noverlap=None)
#     # f_fft_stft, t_fft_stft, _data_fft_stft = signal.stft(data_restore_fft, fs)
#     indx_3 = f_fft_stft <= f_low * 2
#     data_fft_stft[f_fft_stft <= f_low] = 0
#     indx_4 = f_fft_stft >= f_high / 2
#     data_fft_stft[f_fft_stft >= f_high] = 0
#     _, data_restore_fft_stft = signal.istft(data_fft_stft, fs, nperseg=1024, nfft=2048, noverlap=None)
#     data_restore_fft_stft = data_restore_fft_stft[:len(data)]
#
#
#     data_restore_fft = np.concatenate((data_restore_fft, np.zeros(len(data) - len(data_restore_fft))))
#     data_combile_stft_fft = np.add(data_restore_fft, data_restore_stft[:len(data_restore_fft)])/2
#     data_combile_stft_fft_fft_stft = np.add(data_restore_fft_stft, data_restore_stft_fft[:len(data_restore_fft_stft)])/2
#
#     ###### savgol_filter ##########
#     # data_svf = signal.savgol_filter(data_nst, fs // 8, 5, deriv=0)
#     # data_svf -= signal.savgol_filter(data_nst, fs * 4, 1, deriv=0)
#
#     # import statsmodels.api as sm
#     # t = np.arange(len(data_nst[samp_start:samp_end]), dtype=int) / fs
#     # data_svf = sm.nonparametric.lowess(data_nst[samp_start:samp_end], t, frac=1)
#
#     plt.plot(data, 'r', label='raw')
#     # plt.plot(data_restore_stft, 'b', label='stft', alpha=0.6)
#     plt.plot(data_restore_stft_fft, 'b', label='stft_fft')
#     # plt.plot(data_restore_fft, 'y', label='fft', alpha=0.6)
#     plt.plot(data_restore_fft_stft, 'y', label='fft_stft')
#
#     plt.plot(data_bp[samp_start:samp_end], 'g', label='bpf')
#     plt.plot(data_org_bp[samp_start:samp_end], 'k', label='bp_org', alpha=0.8)
#
#     plt.plot(data_svf, 'm', label='raw')
#     # plt.plot(data_restore_fft_stft - data_svf[samp_start:samp_end]/2, 'k', label='raw')
#     # plt.plot(data_combile_stft_fft, '', label='stft_fft')
#     # plt.plot(data_combile_stft_fft_fft_stft, '', label='stft_fft_fft_stft')
#
#     plt.legend()
#     plt.show()
#     # plt.interactive(False)
#
#     a1=0


def main():
    f_low = 0.5
    f_high = 30

    data_org = wf.rdsamp(os.path.join(_mitdb_path, file_name[:-3]))[0][:, 0]
    data_org, _ = resample_sig(data_org, 360, fs)
    data_org_bp = butter_bandpass_filter(data_org, f_low, f_high, fs)

    data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
    data_nst, _ = resample_sig(data_raw, 360, fs)

    data_bp = butter_bandpass_filter(data_nst, f_low, f_high, fs)

    samp_start = int(4.5 * 60 * fs)
    samp_end = int(5.5 * 60 * fs)
    data = data_nst[samp_start:samp_end]

    len_data = len(data)
    # ##### data ==> STFT ==> THR ==> restore ==> FFT ==> THR ==> restore ####
    # f_stft, t_stft, data_stft = signal.stft(data, fs, nperseg=16, nfft=64, noverlap=None)
    # # plt.pcolormesh(t_stft, f_stft, np.abs(data_stft))
    # # plt.show()
    #
    #
    # # f_stft, t_stft, _data_stft = signal.stft(data, fs)
    # indx_1 = f_stft <= f_low
    # data_stft[f_stft <= f_low] = 0
    # indx_2 = f_stft >= f_high
    # data_stft[f_stft >= f_high] = 0
    # _, data_restore_stft = signal.istft(data_stft, fs, nperseg=16, nfft=64, noverlap=None)
    #
    # data_stft_fft = sc.fft.rfft(data_restore_stft[: len(data)])
    # f_stft_fft = sc.fft.rfftfreq(len_data, fs/len_data)
    # data_stft_fft[f_stft_fft <= f_low * 2] = 0
    # data_stft_fft[f_stft_fft > f_high / 2] = 0
    # data_restore_stft_fft = sc.fft.irfft(data_stft_fft)


    ##### data ==> FFT ==> THR ==> restore ==> STFT ==> THR ==> restore ####
    data_fft = sc.fft.rfft(data, len_data)
    # psd = data_fft * np.conj(data_fft) / len_data
    # freq = (1 / (fs * len_data)) * np.arange(len_data)
    # f_fft = sc.fft.rfftfreq(len_data, fs / len_data)# frequency array
    # idxs_half = np.arange(1, np.floor(len_data / 2), dtype=np.int32)
    f, psd = signal.periodogram(data, fs)



    threshold = 75
    psd_idxs = psd < threshold  # array of 0 and 1
    psd_clean = psd * psd_idxs # zero out all the unnecessary powers
    psd_clean[f < f_low] = 0
    psd_clean[f > f_high] = 0

    data_fft_clean = psd_idxs * data_fft  # used to retrieve the signal

    plt.plot(f_fft, np.abs(psd), color='b', lw=0.5, label='PSD noisy')
    # plt.plot(freq[idxs_half], np.abs(psd_clean[idxs_half]), color='r', lw=1, label='PSD clean')
    plt.legend()
    plt.show()

    # data_restore_fft = sc.fft.irfft(data_fft_clean)
    #
    # # data_fft_sc = sc.fft.rfft(data_restore_fft)
    # f_fft = sc.fft.rfftfreq(len_data, fs / len_data)
    # data_fft_clean[f_fft <= f_low / 2] = 0
    # data_fft_clean[f_fft > f_high] = 0
    # data_restore_fft_2 = sc.fft.irfft(data_fft_clean)
    #
    # a=10
    #
    #
    # plt.plot(data, 'r', label='raw')
    #
    # plt.plot(data_bp[samp_start:samp_end], 'y', label='bpf')
    # plt.plot(data_org_bp[samp_start:samp_end], 'k', label='bp_org', alpha=0.8)
    #
    # plt.plot(data_restore_fft, 'm', label='fft')
    # plt.plot(data_restore_fft_2, 'b', label='fft')
    # # plt.plot(data_restore_fft_stft - data_svf[samp_start:samp_end]/2, 'k', label='raw')
    # # plt.plot(data_combile_stft_fft, '', label='stft_fft')
    # # plt.plot(data_combile_stft_fft_fft_stft, '', label='stft_fft_fft_stft')
    #
    # plt.legend()
    # plt.show()
    # plt.interactive(False)

    a1=0

if __name__ == '__main__':
    main()