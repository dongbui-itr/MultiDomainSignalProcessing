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

fs = 250
file = '119'

mitdb_path = '/mnt/Dataset/ECG/PhysionetData/nstdb/'
file_name = file + 'e18'

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


def main():
    data_raw = wf.rdsamp(os.path.join(mitdb_path, file_name))[0][:, 0]
    _data, _ = resample_sig(data_raw, 360, fs)

    len_seg = 60 * fs
    indx = np.arange(0, len_seg, 1)[None, :] + np.arange(0, len(_data) - len_seg, len_seg)[:, None]

    for i in indx[5:]:
        # data = data[3*60*250:7*60*250]
        data = _data[i]
        n_fft = 128
        hop_length = 100 #int(fs * 5)
        win_length = hop_length//5

        n_std_thresh = 1.5

        stft_result = _stft(data, n_fft, hop_length, win_length)
        freqs_stft = np.linspace(1, 0, len(stft_result)) * fs / 2
        # from ssqueezepy import Wavelet, cwt, stft, imshow
        # ikw = dict(abs=1, xticks=np.arange(len(data))/fs, xlabel="Time [sec]", ylabel="Frequency [Hz]")
        # imshow(stft_result, **ikw, yticks=freqs_stft)

        stft_db = _amp_to_db(stft_result)

        # mean_freq_noise = np.mean(stft_result, axis=1)
        # std_freq_noise = np.std(stft_result, axis=1)
        # noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
        #
        # n_grad_freq = 2
        # n_grad_time = 4
        # smoothing_filter = np.outer(
        #     np.concatenate(
        #         [
        #             np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
        #             np.linspace(1, 0, n_grad_freq + 2),
        #         ]
        #     )[1 :-1],
        #     np.concatenate(
        #         [
        #             np.linspace(0, 1, n_grad_time + 1, endpoint=False),
        #             np.linspace(1, 0, n_grad_time + 2),
        #         ]
        #     )[1 :-1],
        # )
        # smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        #
        # db_thresh = np.repeat(
        #     np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        #     np.shape(stft_db)[1],
        #     axis=0,
        # ).T
        # sig_mask = stft_db < db_thresh

        plot_spectrogram(stft_db, 'amplitude spectrogram', data)

    a=10




if __name__ == '__main__':
    main()