import itertools
import webbrowser
import sys
import os
from scipy.signal import butter, lfilter
import numpy as np
from PyQt6 import QtCore
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as Fig
import matplotlib as mpl
from enum import Enum
import config

class Task(Enum):
    REST = 0
    OPEN = 1
    CLOSE = 2
    TRIPOD_OPEN = 3
    TRIPOD_PINCH = 4
    BOTTOM_OPEN = 5
    BOTTOM_CLOSE = 6

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

imgs = (
    resource_path("assets/imgs/rest_label.png"),
    resource_path("assets/imgs/open.png"),
    resource_path("assets/imgs/close.png"),
    resource_path("assets/imgs/tripod_open.png"),
    resource_path("assets/imgs/tripod.png"),
    resource_path("assets/imgs/bottom_open.png"),
    resource_path("assets/imgs/bottom_close.png")
)

pos = {
    "Rest": 0,
    "Open": 1,
    "Close": 2,
    "Tripod Open": 3,
    "Tripod Close": 4,
    "Bottom Open": 5,
    "Bottom Close": 6
}

# Create custom colormap and map the true label with the associated color
cmap = mpl.cm.Spectral
bounds = [0, 1, 2, 3, 4, 5, 6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, clip=True)

colors_list = [cmap.__call__(norm(0)), 
               cmap.__call__(norm(1)), 
               cmap.__call__(norm(2)), 
               cmap.__call__(norm(3)), 
               cmap.__call__(norm(4)), 
               cmap.__call__(norm(5)),]
cmap_discrete = np.array(object=colors_list, dtype=np.float64)

def plot_confusion_matrix(fig, ax, cm, classes,
                         normalize=True,
                         title='Normalized Confusion Matrix',
                         cmap=plt.cm.cividis):
    """
    This function prints and plots the confusion matrix
    Normaliztion can be applied by setting 'normalize=True'.
    """
    
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=45)
    ax.set_yticks(tick_marks, classes, rotation=45)
    
    if normalize:
        cm = cm.copy().astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
    
    thresh = cm.max() / 2.
    for i,j, in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
            horizontalalignment='center',
            color='black' if cm[i, j] > thresh else 'yellow')
    
    fig.set_tight_layout(True)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def documentation():
    # Open documentation about operation
    webbrowser.open(config.resource_path("dev/assets/docs/HoH EMG Hand Documentation Version 4.0.pdf"))

# intersperse a list with a given value... i.e. 0 or rest
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

# Bandpass filter functions
def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def window_rms(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))

# Perform a one hot labeling
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])