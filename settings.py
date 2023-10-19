from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd


# Bandpass filter functions
def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def rolling_rms(x, N):
    return (pd.DataFrame(abs(x) ** 2).rolling(N).mean()) ** 0.5


def window_rms(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


pos = {
    "rest": [1.65, 1.65],
    "open": [3.3, 3.3],
    "close": [0.0, 0.0],
    "tripod": [1.65, 3.3],
    "tripod_open": [3.3, 1.65]
}

pos_list = tuple(pos.values())

pos_rev = {
    "rest": [1.65, 1.65],
    "reverse open": [0.0, 0.0],
    "reverse close": [3.3, 3.3],
    "reverse tripod": [1.65, 0.0],
    "reverse tripod_open": [0.0, 1.65]
}

pos_rev_list = tuple(pos_rev.values())

imgs = (
    "imgs/rest.jpg",
    "imgs/open.jpg",
    "imgs/close.jpg",
    "imgs/tripod.jpg",
    "imgs/tripod_open.jpg",
)

hardware = {'daq': 'H0', 'mindrove': 'H1'}