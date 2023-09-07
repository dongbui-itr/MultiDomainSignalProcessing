import matplotlib.pyplot as plt

import main_2 as lib
import numpy as np
from wfdb.processing import resample_sig

from glob import glob


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


def main():
    dir_path = '/media/dattran88/MegaDataset/Data_Dong/DHYD/Ai-Hourly-Analyzer/'
    dataset = 'study-6000-1'
    FS = 200

    # dir_path = '/media/dattran88/MegaProject/Dong_data/RhythmNet/TestCase/AGD20220331/'
    # dataset = '5ff0b34de9f6c4ba0855eb6d/'
    # FS = 250

    folder = glob(dir_path + dataset + '/*.dat')


    record_config = dict()
    record_config['channels'] = 3
    record_config['gain'] = 655.35875
    record_config['fs'] = 250

    n_fft = 1048
    hop_length = record_config['fs'] * 10
    win_length = hop_length // 4

    for file in folder:
        data_org = get_data_from_dat(file, record_config=record_config, channel=record_config['channels'])
        for ch in range(record_config['channels']):
            ecg, _ = resample_sig(data_org[:, ch], FS, record_config['fs'])

            # ecg = ecg[3 * 60 * 250 :7 * 60 * 250]

            stft_result = lib._stft(ecg, n_fft, hop_length, win_length)
            stft_db = lib._amp_to_db(stft_result)

            ax = plt.subplot(211)
            cax = ax.matshow(
                stft_db,
                origin="lower",
                aspect="auto",
                cmap=plt.cm.seismic,
                vmin=-1 * np.max(np.abs(stft_db)),
                vmax=np.max(np.abs(stft_db)),
            )
            ax = plt.subplot(212)
            ax.plot(ecg)
            plt.show()


if __name__ == '__main__' :
    main()