import itertools
import webbrowser
from scipy.signal import butter, lfilter
import numpy as np
from PyQt6 import QtCore
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as Fig
from matplotlib import cm
from enum import Enum


class Class(Enum):
    REST = 0
    CLOSE = 1
    OPEN = 2
    TRIPOD = 3
    TRIPOD_OPEN = 4


def plot_confusion_matrix(cmi, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix
    Normalization can be applied by setting 'normalize=True'.
    """
    plt.imshow(cmi, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cmi = cmi.astype('float') / cmi.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    thresh = cmi.max() / 2.
    for i, j, in itertools.product(range(cmi.shape[0]), range(cmi.shape[1])):
        plt.text(j, i, cmi[i, j],
                 horizontalalignment='center',
                 color='white' if cmi[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def documentation():
    # Open documentation about operation
    webbrowser.open_new(r'HoH EMG Hand Documentation Version 3.0.pdf')


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


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == QtCore.Qt.Orientation.Vertical:
                return str(self._data.index[section])
