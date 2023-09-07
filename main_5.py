import os
import wfdb as wf
import numpy as np
import librosa
from utils.reprocessing import butter_bandpass_filter

import scipy as sc
import scipy.signal as signal
from wfdb.processing import resample_sig
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import gridspec

fs = 250
file = '119'

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/nstdb/'
file_name = file + 'e06'

# mitdb_path = '/mnt/Dataset/ECG/PhysionetData/mitdb/'
# file_name = '100'


def get_data_from_dat(file, record_config=None, d_type='int16', fromfile=True, channel=3):
    if record_config is None or not bool(record_config):
        num_channel = 3
        gain = 1 #655.35875
    else:
        num_channel = record_config['channels']
        gain = record_config['gain']

    if fromfile:
        sig_data = np.fromfile(file, dtype=d_type)
        sig_data = sig_data / gain
    else:
        sig_data = np.frombuffer(file, dtype=d_type)

    sig_length = len(sig_data) // num_channel
    data = sig_data[: sig_length * num_channel].reshape((sig_length, num_channel), order='C')

    if not fromfile and channel != -1:
        data = np.nan_to_num(data[:, channel])

    return data


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x):
    return librosa.core.db_to_amplitude(x, ref=1.0, amin=1e-20, top_db=80.0)


def plot_spectrogram(signal, title, data):
    ax = plt.subplot(211)
    # fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),

        vmax=np.max(np.abs(signal)),
    )
    plt.subplot(212)
    plt.plot(data)

    # plt.savefig('stft_db.png')
    plt.show()
    # a=10


def plot_statistics_and_filter(
        mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
    ):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")

    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")

    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")

    plt.show()


def stft_basic(x, w, H=8, only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)

    Notebook: C2/C2_STFT-Basic.ipynb

    Args:
        x (np.ndarray): Signal to be transformed
        w (np.ndarray): Window function
        H (int): Hopsize (Default value = 8)
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)

    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """
    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int) + 1
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_win = x[m * H:m * H + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    return X

# def main():
#     data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
#     _data, _ = resample_sig(data_raw, 360, fs)
#
#     len_seg = 35 * fs
#     indx = np.arange(0, len_seg, 1)[None, :] + np.arange(0, len(_data) - len_seg, len_seg)[:, None]
#
#     Fs = fs
#     for i in indx[7:]:
#         # data = data[3*60*250:7*60*250]
#         x = _data[i]
#         H = 512
#         N = H * 2
#         w = np.hanning(N)
#         # X = stft_basic(x, w, H)
#         X = _stft(x, N, H, H)
#         Y = np.abs(X) ** 2
#         eps = np.finfo(float).eps
#         Y_db = 10 * np.log10(Y + eps)
#
#         T_coef = np.arange(X.shape[1]) * H / Fs
#         F_coef = np.arange(X.shape[0]) * Fs / N
#
#         fig = plt.figure(figsize=(8, 5))
#
#         gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2], width_ratios=[100, 2])
#         ax1, ax2, ax3, ax4, ax5, ax6 = [plt.subplot(gs[i]) for i in range(6)]
#
#         t = np.arange(len(x)) / Fs
#         ax1.plot(t, x, c='gray')
#         ax1.set_xlim([min(t), max(t)])
#
#         ax2.set_visible(False)
#
#         left = min(T_coef)
#         right = max(T_coef) + N / Fs
#         lower = min(F_coef)
#         upper = max(F_coef)
#
#         im1 = ax3.imshow(Y, origin='lower', aspect='auto', #cmap='gray_r',
#                          extent=[left, right, lower, upper])
#         im1.set_clim([0, N])
#         ax3.set_ylim([0, 50])
#         ax3.set_ylabel('Frequency (Hz)')
#         cbar = fig.colorbar(im1, cax=ax4)
#         ax4.set_ylabel('Magnitude (linear)', rotation=90)
#
#         im2 = ax5.imshow(Y_db, origin='lower', aspect='auto', #cmap='gray_r',
#                          extent=[left, right, lower, upper])
#         im2.set_clim([-30, 20])
#         ax5.set_ylim([0, 50])
#         ax5.set_xlabel('Time (seconds)')
#         ax5.set_ylabel('Frequency (Hz)')
#         cbar = fig.colorbar(im2, cax=ax6)
#         ax6.set_ylabel('Magnitude (dB)', rotation=90)
#
#         plt.tight_layout()
#         plt.show()
#
#     a=10
#

def main():
    data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
    _data, _ = resample_sig(data_raw, 360, fs)

    len_seg = 35 * fs
    indx = np.arange(0, len_seg, 1)[None, :] + np.arange(0, len(_data) - len_seg, len_seg)[:, None]

    Fs = fs
    for i in indx[8:]:
        # data = data[3*60*250:7*60*250]
        x = _data[i]
        t = np.arange(len(x), dtype=int) / fs
        from PyEMD import EMD, EEMD, Visualisation

        emd = EEMD()
        eIMFs = emd(x)
        imfs, res = emd.get_imfs_and_residue()

        plt.plot(x)
        vis = Visualisation()
        vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
        vis.plot_instant_freq(t, imfs=imfs)
        vis.show()

if __name__ == '__main__':
    main()