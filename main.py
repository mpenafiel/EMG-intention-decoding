from PyQt6 import QtWidgets, QtCore, QtGui, QtSerialPort
import sys
import ctypes
import os

import datetime
import time
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import serial
from keras.layers import Dense
from keras.models import Sequential
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRoveError
from mindrove.data_filter import FilterTypes, DataFilter, NoiseTypes, AggOperations, DetrendOperations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import combinations
from tensorflow import keras
import tensorflow as tf
from keras.optimizers import Adam
import joblib
import random

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme()

import utils
import model_utils
import windows

# Some icons by Yusuke Kamiyamane. Licensed under a Creative Commons Attribution 3.0 License.
# Some icons from freeicons.io

VERSION = '0.5.0'

myappid = utils.resource_path('IDSIcon.ico') # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

def createBoard():
    params = MindRoveInputParams()
    board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD.value, params)
    return board

class IDSCallback(tf.keras.callbacks.Callback):
    def __init__(self, message_signal, epoch_signal, batch_signal):
        tf.keras.callbacks.Callback.__init__(self)
        self.train_err = []
        self.val_err = []
        self.message_signal = message_signal
        self.epoch_signal = epoch_signal
        self.batch_signal = batch_signal
        self.end_training_flag = False
    
    def end_training(self):
        self.end_training_flag = True

    def on_train_begin(self, logs=None):
        msg = 'Starting training'
        self.message_signal.emit(msg)

    def on_epoch_end(self, epoch, logs=None):        
        self.epoch_signal.emit(epoch, logs)
        if self.end_training_flag:
            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        self.batch_signal.emit(batch)
    
    def on_train_end(self, logs=None):
        msg = 'Stopped training'
        self.message_signal.emit(msg)
        
        for key, val in logs.items():
            print(f'{key}: {val}')

class CountdownWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateTime = QtCore.pyqtSignal(int)

    def __init__(self, time_input):
        super().__init__()
        self.timer = time_input

    def run(self):
        while self.timer:
            self.updateTime.emit(self.timer)
            time.sleep(1)
            self.timer -= 1
        self.finished.emit()


class TimerWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateTimer = QtCore.pyqtSignal(int)

    def __init__(self, active_time, inactive_time, tasks):
        super().__init__()
        self.active_timer = active_time
        self.inactive_timer = inactive_time
        self.tasks = tasks
        self.run_flag = True
    
    def interrupt(self):
        self.run_flag = False

    def run(self):
        for i in self.tasks:
            if i == 0:
                timer = self.inactive_timer
            elif i != 0: # Inactive interval
                timer = self.active_timer
            while timer: # Active interval
                if self.run_flag:
                    self.updateTimer.emit(timer)
                    
                    time.sleep(1)
                    timer -= 1
                else:
                    break
        self.finished.emit()


class CollectWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateImg = QtCore.pyqtSignal(int)
    data = QtCore.pyqtSignal(tuple)
    error = QtCore.pyqtSignal(tuple)
    updateProgress = QtCore.pyqtSignal(float)

    def __init__(self, tasks, train_data, train_reps, board, scaler, active_time, inactive_time):
        super().__init__()

        self.tasks = tasks
        self.train_data = train_data
        self.train_reps = train_reps
        self.scaler = scaler
        self.board = board
        self.active_time = active_time
        self.inactive_time = inactive_time
        self.fs = BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD.value) # 500 hz
        self.filter_type = FilterTypes.BUTTERWORTH
        self.ws = 50

        self.run_flag = True

    def stream(self):
        try:
            emg_channels = self.board.get_emg_channels(BoardIds.MINDROVE_WIFI_BOARD.value)
            raw_w_task = np.ndarray(shape=(len(emg_channels) + 1, 0), dtype=np.float64)
            filt_w_task = np.ndarray(shape=(len(emg_channels) + 1, 0), dtype=np.float64)
            
            for i in range(self.train_reps):
                start = time.time()
                k = int(self.tasks[i])
                self.updateImg.emit(k) # Emit task value
                self.board.start_stream()
                if k != 0: # Active Interval
                    time.sleep(self.active_time)
                elif k == 0: #Inactive Interval
                    time.sleep(self.inactive_time)
                self.board.stop_stream()
                data = self.board.get_board_data()
                print(data.shape)
                
                # Processing Data
                gain = 1 # Previously used 12
                RC = 0.045/1000 # Resolution converter to mV
                channels = data[:9,:]*RC*gain # Take channels and scale EMG to mV (gain: 12X ?)
                channels_filtered = channels.copy()
                high_pass_cutoff = 10.0
                for channel in emg_channels:
                    DataFilter.detrend(channels_filtered[channel], detrend_operation=DetrendOperations.LINEAR)
                    DataFilter.perform_highpass(channels_filtered[channel], sampling_rate=self.fs, cutoff=high_pass_cutoff, order=4, filter_type=FilterTypes.BUTTERWORTH, ripple=0)
                    DataFilter.remove_environmental_noise(channels_filtered[channel], sampling_rate=self.fs, noise_type=NoiseTypes.SIXTY)
                
                channels[8, :] = k
                channels_filtered[8, :] = k
                raw_w_task = np.concatenate((raw_w_task, channels), axis=1)
                filt_w_task = np.concatenate((filt_w_task, channels_filtered), axis=1)
                progress = (i + 1) / self.train_reps
                self.updateProgress.emit(progress)
                end = time.time()
                print(f'Computation Time: {end-start} s')
                
                if self.run_flag == False:
                    break
            
            self.board.release_session()

            # Remove first rest phase from dataset
            res = list(np.nonzero(channels_filtered[-1,:]))[0][0] # index of first task
            raw = pd.DataFrame(np.transpose(raw_w_task),
                                columns=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'Task'])
            filtered = pd.DataFrame(np.transpose(filt_w_task),
                                columns=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'Task'])
            
            filtered = filtered.reset_index(drop=True)
            raw = raw.reset_index(drop=True)
            
            smoothed = pd.DataFrame(columns=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'Task'])

            for j in filtered.columns:
                smoothed[j] = utils.window_rms(filtered[j], self.ws)

            smoothed.Task = filtered.Task

            # Prepare channels for normalization
            X = smoothed.iloc[:, :len(smoothed.columns) - 1]  

            if self.scaler is None:
                self.scaler = MinMaxScaler() # Scale all values between 0 and 1
                X_mm = self.scaler.fit_transform(X) # Fit new data
            else:
                X_mm = self.scaler.transform(X) # use pre-loaded weights to scale

            normalized = pd.DataFrame(X_mm, columns=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
            normalized = normalized.assign(Task=filtered.Task)

            print(f'Raw: {len(raw)}, Filtered: {len(filtered)}, Smoothed: {len(smoothed)}, Normalized: {len(normalized)}')

            data = (raw, filtered, smoothed, normalized, self.scaler)  # RAW [0] Filtered [1]... RMS [2] ... Normalized[3]
            if self.run_flag: # only emit data if the run flag is true, i.e. streaming wasn't interrupted
                self.data.emit(data)

            self.updateImg.emit(-1)
            self.finished.emit()

        except ValueError: # in future versions, include the error type and provide user a hint about steps to avoid it.
            self.updateImg.emit(-1)
            self.board.release_session()
            self.finished.emit()
            msg = 'Connection timeout occured. Unable to retrive data from MindRove. Verify it is turned on and connected'
            error = (0, msg)
            self.error.emit(error)
        
        except MindRoveError:
            self.updateImg.emit(-1)
            self.board.release_session()
            self.finished.emit()
            msg = 'Connection timeout occured. Unable to retrive data from MindRove. Verify it is turned on and connected'
            error = (0, msg)
            self.error.emit(error)

    def interrupt(self):
        self.run_flag = False

    def run(self):
        if self.board.is_prepared():
            self.stream()
        else:
            self.board.prepare_session()
            self.stream()


class TrainWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    paramSignal = QtCore.pyqtSignal(tuple)
    dataSignal = QtCore.pyqtSignal(np.ndarray)
    histSignal = QtCore.pyqtSignal(tf.keras.callbacks.History) # History object
    tsneSignal = QtCore.pyqtSignal(np.ndarray)
    cmSignal = QtCore.pyqtSignal(tuple)

    batch_count_signal = QtCore.pyqtSignal(int)
    emit_message_signal = QtCore.pyqtSignal(str)
    epoch_end_signal = QtCore.pyqtSignal(int, dict)
    batch_end_signal = QtCore.pyqtSignal(int)

    def __init__(self, train_data, model, scaler):
        super().__init__()
        self.train_data = train_data
        self.model = model
        self.scaler = scaler
        self.NCC = 2 # NCC = Number of channels to combine
        self.NC = 8 # Number of channels
        self.NFPC = 7 # define the number of features per channel
        self.callbacks = IDSCallback(message_signal=self.emit_message_signal, epoch_signal=self.epoch_end_signal, batch_signal=self.batch_end_signal)

    def run(self):
        # Get channels while excluding task column and convert to numpy array
        X = self.train_data.iloc[:, :- 1].to_numpy()
        tasks = self.train_data.Task.astype(int).to_numpy()
        examples = model_utils.format_examples(X)        

        # Prepare indices of each 2 channels combination
        Indx = np.array(list(combinations(range(self.NC), self.NCC)))   # (28,2)
        # Preallocate memory for number of features
        feat = Indx.shape[0] * self.NFPC + self.NC * self.NFPC

        # Calculate the number of classes, or tasks, being trained, while including the rest phase
        classes = tasks.max() + 1

        labels = utils.perform_labels_windowing(tasks, classes)

        # Remove resting periods from channels
        indices = np.argwhere(labels[:,0] == 0).flatten()
        active_labels = labels[indices,1:]
        active_examples = examples[indices]

        # indices = np.nonzero(self.train_data.Task.astype(int))[0]
        # channels = X[indices]
        # tasks_one_hot = np.delete(tasks_one_hot, 0, axis=1) # Remove first column, which corresponds to rest phase

        if active_examples.shape[0] > active_labels.shape[0]:
            active_examples = active_examples[:active_labels.shape[0],:]
        elif active_examples.shape[0] < active_labels.shape[0]:
            active_labels = active_labels[:active_examples.shape[0]]

        # Split data set into training and testing
        X_train, X_test, y_train, y_test = train_test_split(active_examples, active_labels, test_size=0.30)

        self.batch_count_signal.emit(int(X_train.shape[0]/model_utils.BATCH_SIZE))

        test_set = (X_test, y_test)

        # Classes are defined by all 6 tasks plus rest
        if self.model is None:
            self.model = Sequential([
                tf.keras.Input(shape=(feat,)),
                Dense(250, activation='relu'),
                Dense(64, activation='relu'),
                Dense(units=32, activation='relu'),
                Dense(units=classes-1, activation='softmax')
            ])

        self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(learning_rate=0.01),
                               metrics=['accuracy'])
        
        history = self.model.fit(x=X_train, y=y_train, batch_size=model_utils.BATCH_SIZE, epochs=model_utils.EPOCHS, verbose=0,
                                  validation_data=test_set, validation_batch_size=model_utils.VAL_BATCH_SIZE,
                                  callbacks=[self.callbacks])

        msg = 'Completing final tasks...'
        self.emit_message_signal.emit(msg)

        predictions = self.model.predict(x=active_examples, batch_size=model_utils.BATCH_SIZE, verbose=0)

        predicted_labels = np.argmax(predictions, axis=-1) + 1 # array of predicted labels
        true_labels = np.argmax(active_labels, axis=-1) + 1 # array of true labels

        data = np.column_stack((true_labels, predicted_labels))  # first column [0]->train | second column [1]->predict
        parameters = (self.scaler, self.model)

        # Calculate 3D t-sne for visualization
        tsne3d = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=300).fit_transform(X_test)

        # n_components: Dimension of the embedded space
        # verbose: Verbosity level
        # perpelexity: The perplexity is related to the number of nearest neighbors
        #              that are used in other manifold learning algorithms. Consider 
        #              selecting a value between 5 and 50.
        # n_iter: Maximum number of iterations for the optimization.
        #         Should be at least 250.

        # Create the array of colors based on the true labels
        test_labels = np.argmax(y_test, axis=-1) + 1 # array of true labels

        colors = np.ndarray(shape=(len(test_labels), 4), dtype=np.float64) # Create N x 1 array

        for i in range(6):
            label = i + 1 # Tasks are mapped from 0 to 6
            colors[test_labels==label] = utils.cmap_discrete[i]

        tsne = np.column_stack((tsne3d, colors))

        # Get total number of classes
        unique = np.unique(true_labels)
        classes = list()
        for i in unique:
            _class = utils.Task(i).name
            _class.replace('_', ' ')
            classes.append(_class)

        cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)

        cm_w_class = (cm, classes)

        self.histSignal.emit(history) # contain the history of the model, including loss and accuracy
        self.paramSignal.emit(parameters)  # tuple containing the fitted scaler and model
        self.dataSignal.emit(data)  # numpy 2-D array of poses
        self.tsneSignal.emit(tsne)
        self.cmSignal.emit(cm_w_class)

        self.finished.emit()


class TestWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateImg = QtCore.pyqtSignal(int)
    validateTask = QtCore.pyqtSignal(list)
    predictions = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(tuple)

    def __init__(self, tasks, selected_data, train_reps, model, scaler, board, serial_obj, active_time, inactive_time, fixed_mode):
        super().__init__()

        self.tasks = tasks
        self.channels = selected_data  # The selected channels feed into the model
        self.train_reps = train_reps
        self.model = model
        self.scaler = scaler
        self.board = board
        self.serial_obj = serial_obj
        self.active_time = active_time
        self.inactive_time = inactive_time
        self.fixed_mode = fixed_mode
        # Constants
        self.fs = BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD.value)
        self.filter_type = FilterTypes.BUTTERWORTH
        self.ws = 50

    def test(self):
        try:
            if self.channels is not None:
                test_data = pd.DataFrame(columns=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
                for i in test_data.columns:
                    if i not in self.channels:
                        test_data.drop(columns=[i])

            emg_channels = self.board.get_emg_channels(BoardIds.MINDROVE_WIFI_BOARD.value)
            tasks = list()
            true_tasks = list()

            predicted_labels = np.ndarray(shape=(0,), dtype=np.int64)

            for i in self.tasks:
                if self.fixed_mode:
                    self.updateImg.emit(i)

                    # pseudo tasks are performed here
                    if i != 0:
                        byte_command = bytes('0', 'utf-8')
                        self.serial_obj.write(byte_command) # Command to return home
                        time.sleep(self.active_time)

                        byte_command = bytes(str(i), 'utf-8')
                        print(f'Predicted Task: {i}')
                        self.serial_obj.write(byte_command)
                        time.sleep(self.active_time)

                        byte_command = bytes('0', 'utf-8')
                        self.serial_obj.write(byte_command) # Command to return home
                    elif i == 0:
                        time.sleep(self.inactive_time)
                else:
                    start = time.time()
                    self.updateImg.emit(i)
                    
                    if i != 0: # Active Interval.. default is 3 seconds
                        print('Starting new task')
                        byte_command = bytes('0', 'utf-8')
                        self.serial_obj.write(byte_command) # Command to return home
                        self.board.start_stream()
                        time.sleep(self.active_time)
                        self.board.stop_stream()
                        data = self.board.get_board_data()
                        print(f'Data Shape: {data.shape}')
                        # Processing Data
                        gain = 1 # Previously used 12
                        RC = 0.045/1000 # Resolution converter to mV
                        channels = data[:8,:]*RC*gain # Take channels and scale EMG to mV (gain: 12X ?)
                        channels_filtered = channels.copy()
                        for channel in emg_channels:
                            DataFilter.detrend(channels_filtered[channel], detrend_operation=DetrendOperations.LINEAR)
                            DataFilter.perform_highpass(channels_filtered[channel], sampling_rate=self.fs, cutoff=50.0, order=4, filter_type=FilterTypes.BUTTERWORTH, ripple=0)
                            DataFilter.remove_environmental_noise(channels_filtered[channel], sampling_rate=self.fs, noise_type=NoiseTypes.SIXTY)

                        X = channels_filtered[:8,:]

                        examples = model_utils.format_examples(X)

                        Y = self.model.predict(examples)

                        labels = np.argmax(Y, axis=-1)  # Array of predicted labels

                        predicted_labels = np.concatenate([predicted_labels, labels], dtype=np.int64)

                        predicted_array = np.bincount(labels)
                        print(f'Prediction Array: {predicted_array}')

                        predicted_label = predicted_array.argmax() + 1

                        print(f'Predicted Task: {predicted_label}')

                        byte_command = bytes(str(predicted_label), 'utf-8')
                        self.serial_obj.write(byte_command)
                        time.sleep(self.active_time)

                        tasks.append(predicted_label)
                        true_tasks.append(i)

                        byte_command = bytes('0', 'utf-8')
                        self.serial_obj.write(byte_command) # Command to return home
                        end = time.time()
                    elif i == 0: #Inactive Interval
                        self.board.start_stream()
                        time.sleep(self.inactive_time)
                        self.board.stop_stream()
                        discard = self.board.get_board_data()
                        end = time.time()
                    print(f'Computation time: {end-start}')

            print(f'Predicted Tasks: {tasks}')
            print(f'True Tasks: {true_tasks}')
            self.updateImg.emit(-1)
            self.predictions.emit(predicted_labels)

            self.board.release_session()
        except MindRoveError:
            msg = 'Connection timeout occured. Unable to retrive data from MindRove. Verify it is turned on and connected.'
            self.updateImg.emit(-1)
            self.board.release_session()
            self.finished.emit()
            error = (0, msg)
            self.error.emit(error)
        self.finished.emit()

    def run(self):
        if self.board.is_prepared():
            self.test()
        else:
            self.board.prepare_session()
            self.test()

class PoseApp(QtWidgets.QMainWindow): 
    train_data = None
    selected_data = None  # specify the selected channels and samples for training

    def __init__(self):
        # Call the Parent constructor
        super().__init__()

        # Data Constructors
        self.raw_data = None
        self.filt_data = None
        self.rms_data = None

        # Worker and Thread Constructors
        self.testWorker = None
        self.testThread = None
        self.trainWorker = None
        self.trainThread = None
        self.collectWorker = None
        self.collectThread = None
        self.countdownWorker = None
        self.countdownThread = None
        self.timerWorker = None
        self.timerThread = None

        # Paths
        self.subject_path = None
        self.data_path = None
        self.model_path = None

        self.pixmap = QtGui.QPixmap()

        self.data_time = None
        self.tasks = []
        self.train_reps = len(self.tasks)
        self.active_time = 3  # Default active period in seconds
        self.inactive_time = self.active_time # Default is to have same interval for tasks and rest
        self.timer = self.active_time
        self.test_data = None
        self.scaler = None
        self.model = None
        self.table = None
        self.serial_obj = None
        self.board = createBoard()
        self.fs = BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD.value)
        self.ports = None

        self.setWindowIcon(QtGui.QIcon('IDSIcon.png'))
        self.setWindowTitle("Intention Detection")
        self.resize(1200, 675)

        tabs = QtWidgets.QTabWidget(self)
        self.mainTab = windows.MainUi(self)
        tabs.addTab(self.mainTab, "Main")
        self.analysisTab = windows.AnalysisUI(self)
        tabs.addTab(self.analysisTab, "Analysis")

        central_widget_layout = QtWidgets.QVBoxLayout()
        central_widget_layout.setContentsMargins(0, 0, 0, 0)
        central_widget_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        central_widget_layout.addWidget(tabs)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(central_widget_layout)

        self.setCentralWidget(tabs)

        self._createActions()
        self._createMenuBar()
        self._createStatusBar()
        self._connectActions()

        self.mainTab.collect.clicked.connect(self.begin_collecting_data)
        self.mainTab.train.clicked.connect(self.begin_training_data)
        self.mainTab.test.clicked.connect(self.begin_testing_data)
        self.mainTab.detect.clicked.connect(self.detect_available_ports)
        self.mainTab.connect_port.clicked.connect(self.connect_port)
        self.mainTab.interrupt.clicked.connect(self.interrupt_data)
        self.mainTab.check_group.idClicked.connect(self.update_channels_from_main_tab)
        self.analysisTab.check_group.idClicked.connect(self.update_channels_from_analysis_tab)

        self.channel_plots = {'CH1': self.analysisTab.CH1,
                            'CH2': self.analysisTab.CH2,
                            'CH3': self.analysisTab.CH3,
                            'CH4': self.analysisTab.CH4,
                            'CH5': self.analysisTab.CH5,
                            'CH6': self.analysisTab.CH6,
                            'CH7': self.analysisTab.CH7,
                            'CH8': self.analysisTab.CH8,
                            'Task': self.analysisTab.TASK}
        
        self.channel_task = {'CH1': self.analysisTab._ch1_task,
                             'CH2': self.analysisTab._ch2_task,
                             'CH3': self.analysisTab._ch3_task,
                             'CH4': self.analysisTab._ch4_task,
                             'CH5': self.analysisTab._ch5_task,
                             'CH6': self.analysisTab._ch6_task,
                             'CH7': self.analysisTab._ch7_task,
                             'CH8': self.analysisTab._ch8_task,}

        self.channel_checkbox = {'CH1': (self.mainTab.ch1_main, self.analysisTab.ch1_check),
                                'CH2': (self.mainTab.ch2_main, self.analysisTab.ch2_check),
                                'CH3': (self.mainTab.ch3_main, self.analysisTab.ch3_check),
                                'CH4': (self.mainTab.ch4_main, self.analysisTab.ch4_check),
                                'CH5': (self.mainTab.ch5_main, self.analysisTab.ch5_check),
                                'CH6': (self.mainTab.ch6_main, self.analysisTab.ch6_check),
                                'CH7': (self.mainTab.ch7_main, self.analysisTab.ch7_check),
                                'CH8': (self.mainTab.ch8_main, self.analysisTab.ch8_check)}

        self.analysisTab.display_all.clicked.connect(self.display_all_channels)
        self.analysisTab.clear_all.clicked.connect(self.clear_channel_plots)

        self.analysisTab.clear_shift.clicked.connect(self.clear_shift)
        self.analysisTab.layer_task.stateChanged.connect(self.check_overlay_state)
        self.analysisTab.set_shift.clicked.connect(self.set_shift)

        self.analysisTab.data_group.buttonClicked.connect(self.set_datatype)
        
        # Center window in center of main screen
        center = QtGui.QScreen.availableGeometry(QtWidgets.QApplication.primaryScreen()).center()
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

        self.show()

    # Override method for class MainWindow
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        QtWidgets.QMainWindow.resizeEvent(self, event)
        w = self.mainTab.task_img.width()
        h = self.mainTab.task_img.height()
        self.mainTab.task_img.setPixmap(self.pixmap.scaled(w,h, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def _updateViews(self, p1, p2):
            ## view has resized; update auxiliary views to match
            p2.setGeometry(p1.vb.sceneBoundingRect())

            ## need to re-update linked axes since this was called
            ## incorrectly while views had different shapes.
            ## (probably this should be handled in ViewBox.resizeEvent)
            p2.linkedViewChanged(p1.vb, p2.XAxis)

    def _createActions(self):
        # Creating actions using the second constructor
        self.loadSubjectAction = QtGui.QAction("Open Subject Folder...", self)
        self.loadDataAction = QtGui.QAction("Open &Data...", self)
        self.loadModelAction = QtGui.QAction("Open &Model...", self)

        self.saveDataAction = QtGui.QAction("Save Data", self)
        self.saveDataAsAction = QtGui.QAction("Save Data as...", self)
        self.saveModelAction = QtGui.QAction("Save Model", self)
        self.saveModelAsAction = QtGui.QAction("Save Model as...", self)

        self.clearDataAction = QtGui.QAction("Clear Data", self)
        self.clearModelAction = QtGui.QAction("Clear Model", self)
        self.clearAllAction = QtGui.QAction("Clear All", self)
        self.exitAction = QtGui.QAction("&Exit", self)

        self.newTasksAction = QtGui.QAction("&New Task Sequence...", self)
        self.customTasksAction = QtGui.QAction("&Custom Task Sequence...", self)
        self.randTasksAction = QtGui.QAction("&Randomized Task Sequence...", self)
        self.createTrainingAction = QtGui.QAction("Training Parameters", self)
        self.changeTimeAction = QtGui.QAction("Set Time Interval", self)

        self.documentationAction = QtGui.QAction("&Documentation", self)
        self.exampleTasksAction = QtGui.QAction("Getting Started", self)
        self.aboutAction = QtGui.QAction("&About", self)
        self.feedbackAction = QtGui.QAction('Send &Feedback', self)

    def _connectActions(self):
        # Connect File actions
        self.loadSubjectAction.triggered.connect(self.load_create_subject_folder)
        self.loadDataAction.triggered.connect(self.load_data)
        self.loadModelAction.triggered.connect(self.load_model)
        self.saveDataAction.triggered.connect(lambda: self.save_data_from_menu(saveAs=False))
        self.saveDataAsAction.triggered.connect(lambda: self.save_data_from_menu(saveAs=True))
        self.saveModelAction.triggered.connect(lambda: self.save_model_from_menu(saveAs=False))
        self.saveModelAsAction.triggered.connect(lambda: self.save_model_from_menu(saveAs=True))
        self.clearDataAction.triggered.connect(self.clear_data)
        self.clearModelAction.triggered.connect(self.clear_model)
        self.clearAllAction.triggered.connect(self.clear_all)
        self.exitAction.triggered.connect(self.close)

        # Connect Tools actions
        self.newTasksAction.triggered.connect(self.create_new_task_sequence)
        self.customTasksAction.triggered.connect(self.create_custom_task_sequence)
        self.randTasksAction.triggered.connect(self.create_random_task_sequence)
        self.createTrainingAction.triggered.connect(self.create_training_parameters)
        self.changeTimeAction.triggered.connect(self.change_time)

        # Connect Icon actions
        self.mainTab.subject_icon.clicked.connect(self.load_create_subject_folder)
        self.mainTab.data_icon.clicked.connect(self.load_data)
        self.mainTab.model_icon.clicked.connect(self.load_model)
        self.mainTab.time_icon.clicked.connect(self.change_time)
        self.mainTab.mindrove_icon.clicked.connect(self.verify_mindrove_status)

        # Connect Help actions
        self.documentationAction.triggered.connect(utils.documentation)
        self.exampleTasksAction.triggered.connect(self.example_task_sequence)
        self.aboutAction.triggered.connect(self.about)
        self.feedbackAction.triggered.connect(self.send_feedback)

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = menuBar.addMenu("File")
        menuBar.addMenu(fileMenu)
        openMenu = fileMenu.addMenu("Open")
        openMenu.addAction(self.loadSubjectAction)
        openMenu.addAction(self.loadDataAction)
        openMenu.addAction(self.loadModelAction)
        # Save menu
        saveMenu = fileMenu.addMenu("Save")
        dataMenu = saveMenu.addMenu("Data")
        dataMenu.addAction(self.saveDataAction)
        dataMenu.addAction(self.saveDataAsAction)
        modelMenu = saveMenu.addMenu("Model")
        modelMenu.addAction(self.saveModelAction)
        modelMenu.addAction(self.saveModelAsAction)
        # Clear menu
        clearMenu = fileMenu.addMenu("Clear")
        clearMenu.addAction(self.clearDataAction)
        clearMenu.addAction(self.clearModelAction)
        clearMenu.addAction(self.clearAllAction)
        # Exit
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Tools menu
        toolMenu = menuBar.addMenu("Tools")
        tasksMenu = toolMenu.addMenu("Create Task Sequence")
        tasksMenu.addAction(self.newTasksAction)
        tasksMenu.addAction(self.customTasksAction)
        tasksMenu.addAction(self.randTasksAction)

        toolMenu.addAction(self.createTrainingAction)
        toolMenu.addAction(self.changeTimeAction)

        # Help menu
        helpMenu = menuBar.addMenu("Help")
        helpMenu.addAction(self.documentationAction)
        helpMenu.addAction(self.exampleTasksAction)
        helpMenu.addAction(self.aboutAction)
        helpMenu.addAction(self.feedbackAction)

    def _createStatusBar(self):
        self.statusbar = self.statusBar() # #44464d;
        # Adding a temporary message
        self.statusbar.showMessage("Ready", 3000)
        self.connection_status = QtWidgets.QLabel(self.current_port_status())
        self.statusbar.addPermanentWidget(self.connection_status)

    def about(self):
        message = f'Version: {VERSION}'
        QtWidgets.QMessageBox.information(self, 'Intention Detection System', message)

    def check_overlay_state(self):
        if self.analysisTab.layer_task.isChecked():
            task = self.train_data.Task
            t2 = (task.index / self.fs).to_numpy().flatten()

            for channel_name in self.train_data.iloc[:,:-1]:
                channel = self.channel_plots.get(channel_name)
                channel.showAxis('right', show=True)
                p1 = channel.getPlotItem()  # gets the plot item from the plot widget
                p2 = self.channel_task.get(channel_name) # get the viewbox module
                self._updateViews(p1, p2)
                p1.vb.sigResized.connect(lambda: self._updateViews(p1, p2))

                p2.addItem(pg.PlotCurveItem(x= t2, y=task.to_numpy().flatten(), pen='r'))

        else:
            for channel_name in self.train_data.iloc[:,:-1]:
                channel = self.channel_plots.get(channel_name)
                channel.showAxis('right', show=False)
                p1 = channel.getPlotItem()  # gets the plot item from the plot widget
                p2 = self.channel_task.get(channel_name) # get the viewbox module
                p2.clear()

    def clear_all(self):
        self.clear_data()
        self.clear_model()

    def clear_data(self):
        self.train_data = np.ndarray(shape=(0, 9), dtype=np.int64)
        self.mainTab.data_status.clear()
        self.mainTab.data_status.setToolTip(str())
        self.analysisTab.table.setModel(None)
        self.clear_all_plots()

        # Reset buttons
        for channel_name in self.channel_checkbox.values():
            channel_name[0].setEnabled(False)
            channel_name[0].setChecked(False)
            channel_name[1].setEnabled(False)
            channel_name[1].setChecked(False)

        self.analysisTab.display_all.setEnabled(False)
        self.analysisTab.clear_all.setEnabled(False)

        self.analysisTab.layer_task.setEnabled(False)
        self.analysisTab.shift.setEnabled(False)
        self.analysisTab.clear_shift.setEnabled(False)
        self.analysisTab.set_shift.setEnabled(False)
        self.analysisTab.total_samples.setText('Total Samples: 0')
        self.analysisTab.down_samples.setText('Downsampled: 0')
        self.analysisTab.layer_task.setChecked(False)

        self.analysisTab.raw.setEnabled(False)
        self.analysisTab.filt.setEnabled(False)
        self.analysisTab.rms.setEnabled(False)
        self.analysisTab.norm.setEnabled(False)

    def clear_all_plots(self):
        plots = self.channel_plots.values()
        viewboxes = self.channel_task.values()
        for plot_widget in plots:
            plot_widget.clear()
            plot_widget.showAxis('right', show=False)
        for viewbox in viewboxes:
            viewbox.clear()            

    def clear_channel_plots(self):
        for channel in self.channel_checkbox.values():
            channel[0].setChecked(False)
            channel[1].setChecked(False)

        plots = self.channel_plots.values()
        for plot_item in plots:
            if plot_item != self.analysisTab.TASK:
                plot_item.clear()

    def clear_model(self):
        self.model = None
        self.mainTab.model_status.clear()
        self.mainTab.model_status.setToolTip(str())
    
    def clear_shift(self):
        self.selected_data = self.train_data
        self.analysisTab.shift.setValue(0)
        self.analysisTab.down_samples.setText(f'Downsampled: {int(self.analysisTab.shift.value())} samples')
        self.display_data(self.selected_data)

    def reset_task_instances_ui(self):
        self.mainTab.task_open.setText(str(0)) # set all values to 0 before setting new values
        self.mainTab.task_close.setText(str(0))
        self.mainTab.task_tripod_open.setText(str(0))
        self.mainTab.task_tripod_pinch.setText(str(0))
        self.mainTab.task_bottom_open.setText(str(0))
        self.mainTab.task_bottom_close.setText(str(0))

        unique , counts = np.unique(self.tasks, return_counts=True)
        task_instances = dict(zip(unique, counts))

        for key, val in task_instances.items():
            if key == utils.Task.OPEN.value:
                self.mainTab.task_open.setText(str(val))
            elif key == utils.Task.CLOSE.value:
                self.mainTab.task_close.setText(str(val))
            elif key == utils.Task.TRIPOD_OPEN.value:
                self.mainTab.task_tripod_open.setText(str(val))
            elif key == utils.Task.TRIPOD_PINCH.value:
                self.mainTab.task_tripod_pinch.setText(str(val))
            elif key == utils.Task.BOTTOM_OPEN.value:
                self.mainTab.task_bottom_open.setText(str(val))
            elif key == utils.Task.BOTTOM_CLOSE.value:
                self.mainTab.task_bottom_close.setText(str(val))

    def create_new_task_sequence(self):
        # Generate a new task sequence by choosing the number of tasks
        dlg = windows.NewTaskDialog(self)

        if dlg.exec():
            self.tasks = []
            tasks = []
            for key, val in dlg.tasks.items():
                for i in range(key.value()):
                    tasks.append(val)

            if dlg.shuffle.isChecked():
                tasks = random.sample(tasks, len(tasks)) # randomly shuffle the tasks
            
            self.tasks = [utils.Task.REST.value] + utils.intersperse(tasks, utils.Task.REST.value) # add a rest phase inbetween each task
            self.train_reps = len(self.tasks)
            self.reset_task_instances_ui()
        else:
            message = 'User cancelled'
            self.statusbar.showMessage(message, 3000)

    def create_custom_task_sequence(self):
        # Generate a new tasks with any task assignment length between 0 and 20

        while True:
            new_tasks, ok = QtWidgets.QInputDialog.getText(
                self, 'New Task Sequence', 'Enter new task sequence:')
            
            if ok:
                new_tasks = new_tasks.split()
                for i in range(len(new_tasks)):
                    if new_tasks[i].isdigit():
                        new_tasks[i] = int(new_tasks[i])

                if all([isinstance(item, int) for item in new_tasks]):
                    if all([int(item) >= 1 for item in new_tasks]) and all([int(item) <= 6 for item in new_tasks]):
                        if len(new_tasks) <= 30 and len(new_tasks) > 0:
                            self.tasks = [utils.Task.REST.value] + utils.intersperse(new_tasks, utils.Task.REST.value) # add a rest phase inbetween each task
                            self.train_reps = len(self.tasks)

                            self.reset_task_instances_ui()
                            break
                        else:
                            message = 'Length should be between 1 and 30'
                            QtWidgets.QMessageBox.information(self, 'Help', message)
                            continue
                    else:
                        message = 'Integers should be between 1 and 6'
                        QtWidgets.QMessageBox.information(self, 'Help', message)
                        continue
                else:
                    message = 'Create input as integers separated by space'
                    QtWidgets.QMessageBox.information(self, 'Help', message)
                    continue
            else:
                break

    def create_random_task_sequence(self):
        # Generate a randomized task sequence of any length between 0 and 30
        dlg = windows.RandTaskDialog(self)

        if dlg.exec():
            pass
            if dlg.set_length.isChecked():
                length = dlg.length.value()
            else:
                length = np.random.randint(4, high=31)

            new_tasks = np.random.randint(low=1, high=7, size=(1, length)) # low is inclusive, high is exclusive
            new_tasks = new_tasks.tolist()
            new_tasks = new_tasks[0]  # obtain the list

            self.tasks = [utils.Task.REST.value] + utils.intersperse(new_tasks, utils.Task.REST.value) # add a rest phase inbetween each task
            self.train_reps = len(self.tasks)

            self.reset_task_instances_ui()
        else:
            message = 'User cancelled'
            self.statusbar.showMessage(message, 3000)

    def create_training_parameters(self):
        dlg = windows.ParameterDialog(self)

        if dlg.exec():
            self.tasks = []
            tasks = []
            for key, val in dlg.tasks.items():
                if key.isChecked():
                    i = dlg.repetitions.value()
                    while i: # append all repetitions of task
                        tasks.append(val)
                        i = i -1

            shuffled_tasks = random.sample(tasks, len(tasks)) # randomly shuffle the tasks
            self.tasks = [utils.Task.REST.value] + utils.intersperse(shuffled_tasks, utils.Task.REST.value) # add a rest phase inbetween each task
            self.train_reps = len(self.tasks)
            self.reset_task_instances_ui()
        else:
            message = 'User cancelled'
            self.statusbar.showMessage(message, 3000)

    def detect_available_ports(self):
        self.mainTab.port_select.clear()
        self.ports = QtSerialPort.QSerialPortInfo().availablePorts()

        if len(self.ports) != 0:
            self.mainTab.port_select.clear()
            for port in self.ports:
                self.mainTab.port_select.addItem(port.portName())
        else:
            self.mainTab.port_select.addItem('Not Available')
            self.mainTab.port_status.setText('Not Connected')

    #  function to connect to microcontroller... configure so that baudrate match and are optimized
    def connect_port(self):
        if self.mainTab.port_select.currentText() == 'Not Available':
            message = 'No port was detected'
            QtWidgets.QMessageBox.warning(self, 'Warning', message)
        elif self.serial_obj is None:
            self.serial_obj = serial.Serial(self.mainTab.port_select.currentText(), baudrate=9600, timeout=5,
                                            write_timeout=5)
            self.connection_status.setText(self.current_port_status())
            self.mainTab.port_status.setText(self.serial_obj.port)
        elif self.serial_obj.is_open:
            if self.serial_obj.port == self.mainTab.port_select.currentText():
                message = f'{self.mainTab.port_select.currentText()} is already connected'
                QtWidgets.QMessageBox.information(self, 'Help', message)
            else:
                self.serial_obj.close()
                self.serial_obj = serial.Serial(self.mainTab.port_select.currentText(), baudrate=9600, timeout=5,
                                                write_timeout=5)
                self.connection_status.setText(self.current_port_status())
                self.mainTab.port_status.setText(self.serial_obj.port)

    def current_port_status(self):
        if self.serial_obj is None:
            x = 'Device not connected'
        else:
            x = f'Device is connected to {self.serial_obj.port}'
        return x

    def change_time(self):
        dlg = windows.TimeIntervalDialog(self)

        if dlg.exec():
            if dlg.assign_inactive.isChecked():
                self.active_time = dlg.active.value()
                self.inactive_time = dlg.inactive.value()
                self.mainTab.time_interval.setText(f'Active: {self.active_time}, Inactive: {self.inactive_time}')
            else:
                self.active_time = dlg.active.value()
                self.inactive_time = self.active_time
                self.mainTab.time_interval.setText(f'Active/Inactive: {self.active_time}')

    def emit_error_message(self, event):
        error_type = event[0]
        msg = event[1]
        if error_type == 0:
            self.mainTab.mindrove_status.setText("Not Connected")
        QtWidgets.QMessageBox.warning(self, 'Warning', msg)

    def example_task_sequence(self):
        self.helpWin = windows.ExampleWin(self)
        self.helpWin.show()

    def interrupt_data(self):
        self.collectWorker.interrupt()
        self.timerWorker.interrupt()

    def load_create_subject_folder(self):
        subject_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Subject",
                                                                  options=QtWidgets.QFileDialog.Option.ShowDirsOnly)
        if subject_path:
            subject_name = os.path.basename(subject_path)
            self.mainTab.subject.setText(subject_name)
            self.subject_path = subject_path
            self.mainTab.subject.setToolTip(subject_path)
        else:
            message = 'User Cancelled'
            self.statusbar.showMessage(message, 3000)

    def load_data(self):
        try:
            filt = "CSV files(*.csv);;XML files(*.xml);;Text files (*.txt)"
            if self.subject_path:
                fname, selFilter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', self.subject_path, filt)
            else:
                fname, selFilter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', filt)
            if fname:
                self.train_data = pd.read_csv(fname)
                self.data_path = fname
                self.selected_data = self.train_data
                message = f'{fname} successfully uploaded!'
                self.mainTab.data_status.setToolTip(str(fname))
                file_name = os.path.basename(fname)
                self.statusbar.showMessage(message, 3000)
                self.mainTab.data_status.setText(file_name)
                self.analysisTab.raw.setEnabled(True)
                self.analysisTab.filt.setEnabled(True)
                self.analysisTab.rms.setEnabled(True)
                self.analysisTab.norm.setEnabled(True)
                self.analysisTab.norm.setChecked(True)  # default display is normalized data
                # Display Shift Functions
                self.analysisTab.clear_shift.setEnabled(True)
                self.analysisTab.set_shift.setEnabled(True)
                self.analysisTab.layer_task.setEnabled(True)
                self.analysisTab.shift.setEnabled(True)
                self.analysisTab.total_samples.setText(f'Total Samples: {self.train_data.shape[0]}')

                print(f"Train Data Shape: {self.train_data.shape}")
                self.display_data(self.train_data)
            else:
                message = 'User Cancelled'
                self.statusbar.showMessage(message, 3000)
        except OSError:
            message = 'Unable to open file'
            self.statusbar.showMessage(message, 3000)

    def load_model(self):
        filt = "Keras files(*.keras);;H5 files(*.h5);;Tensorflow files (*.tf)"
        if self.subject_path:
            (model_path, selFilter) = QtWidgets.QFileDialog.getOpenFileName(self, "Open Model", self.subject_path, filter=filt)
        else:
            (model_path, selFilter) = QtWidgets.QFileDialog.getOpenFileName(self, "Open Model", 'c:\\', filter=filt)
        try:
            if model_path:
                self.mainTab.model_status.setToolTip(model_path)
                self.model_path = model_path
                self.model_path.replace("\\", "/")
                self.model = tf.keras.models.load_model(model_path)
                scaler_path = model_path[:-6] + "_scaler.save"
                self.scaler = joblib.load(scaler_path)
                message = f'{model_path} successfully uploaded!'
                folder_name = os.path.basename(model_path)
                self.statusbar.showMessage(message, 3000)
                self.mainTab.model_status.setText(folder_name)
                for name, channel in self.channel_checkbox.items():
                    channel_main = channel[0]
                    channel_main.setEnabled(True)
                    channel_main.setChecked(True)

                    channel_analysis = channel[1]
                    channel_analysis.setEnabled(True)
                    channel_analysis.setChecked(True)

                self.selected_data = pd.DataFrame(columns=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
            else:
                message = 'User Cancelled'
                self.statusbar.showMessage(message, 3000)
        except ValueError:
            file_name = os.path.basename(model_path)
            message = f'Unable to rebuild {file_name}'
            self.statusbar.showMessage(message, 3000)
        except OSError:
            file_name = os.path.basename(model_path)
            message = f'Unable to open {file_name}'
            self.statusbar.showMessage(message, 3000)

    def send_feedback(self):
        import webbrowser

        webbrowser.open("https://docs.google.com/forms/d/e/1FAIpQLSe2TxeGHhEYwc9NZYuw-jtJWkVD9ftkLbaGcbPRQQGawXs7Ug/viewform?usp=sf_link")

    def close_event(self, event):
        if self.board.is_prepared():
            close = QtWidgets.QMessageBox.question(self,
                                                   "QUIT",
                                                   "ARE YOU SURE? Quitting will interrupt data collection",
                                                   QtWidgets.QMessageBox.StandardButton.Yes |
                                                   QtWidgets.QMessageBox.StandardButton.No)
            if close == QtWidgets.QMessageBox.StandardButton.Yes:  # add boolean statement for threads
                if self.serial_obj is not None and self.serial_obj.is_open:
                    self.serial_obj.close()
                try:
                    self.board.release_session()
                except:
                    pass # add some notification if error is raised, i.e. the board is not connected
                event.accept()
            elif close == QtWidgets.QMessageBox.StandardButton.No:
                event.ignore()
        else:
            if self.serial_obj is not None and self.serial_obj.is_open:
                self.serial_obj.close()
            event.accept()

    def update_channels_from_main_tab(self, id_):
        checkbox = self.analysisTab.check_group.button(id_)
        channel_name = self.mainTab.check_group.button(id_).text()
        if checkbox.isChecked():
            checkbox.setChecked(False)
        else:
            checkbox.setChecked(True)
        self.update_plot(checkbox)

        self.selected_data = self.train_data  # create a copy of the original data
        for key, value in self.channel_checkbox.items():
            if not value[0].isChecked():
                self.selected_data = self.selected_data.drop(columns=[key])

    def update_channels_from_analysis_tab(self, id_):
        checkbox = self.mainTab.check_group.button(id_)
        channel_name = self.analysisTab.check_group.button(id_).text()

        if checkbox.isChecked():
            checkbox.setChecked(False)
        else:
            checkbox.setChecked(True)
        plot = self.channel_plots.get(channel_name)
        self.update_plot(checkbox)

        self.selected_data = self.train_data  # create a copy of the original data
        for key, value in self.channel_checkbox.items():
            if not value[1].isChecked():
                self.selected_data = self.selected_data.drop(columns=[key])

    def display_data(self, data):
        self.clear_all_plots()
        t2 = (data.index / self.fs).to_numpy().flatten()
        task = data.Task                

        for count, channel_name in enumerate(data):
            channel = self.channel_plots.get(channel_name)  # gets the plot
            channel.plot(x = t2, y = data[channel_name].to_numpy())

            if channel_name != 'Task':  # ignore the channel containing pose
                if self.analysisTab.layer_task.isChecked():
                    channel.showAxis('right', show=True)
                    p1 = channel.getPlotItem()  # gets the plot item from the plot widget
                    p2 = self.channel_task.get(channel_name) # get the viewbox module
                    self._updateViews(p1, p2)
                    p1.vb.sigResized.connect(lambda: self._updateViews(p1, p2))
                    p2.addItem(pg.PlotCurveItem(x= t2, y=task.to_numpy().flatten(), pen='r'))

                channel_check = self.channel_checkbox.get(channel_name)
                channel_check[0].setEnabled(True)
                channel_check[0].setChecked(True)
                channel_check[1].setEnabled(True)
                channel_check[1].setChecked(True)

        self.table = windows.TableModel(data)
        self.analysisTab.table.setModel(self.table)

    def update_plot(self, checkbox):
        if len(self.train_data) > 0 or type(self.train_data) is pd.DataFrame:
            channel = self.channel_plots.get(checkbox.text())  # get the plot object from the name of the checkbox

            if checkbox.isChecked():
                t2 = self.train_data.index / self.fs
                channel.plot(t2, self.train_data[checkbox.text()].to_numpy())  # plot the data one specified plot
            else:
                channel.clear()  # clear the data from the specified plot

    def display_all_channels(self):
        for i in self.channel_checkbox.values():
            i[0].setChecked(True)
            i[1].setChecked(True)
            self.update_plot(i[1])

    def display_confusion_matrix(self, event):
        cm = event[0]
        classes = event[1]
        utils.plot_confusion_matrix(fig=self.analysisTab._cm_fig, ax=self.analysisTab._cm_ax, cm=cm, classes=classes)
        
    
    def display_model_history(self, event):
        history = event

        self.analysisTab._mal_ax.plot(history.history['accuracy'], linestyle='solid', color='blue')
        self.analysisTab._mal_ax.plot(history.history['val_accuracy'], linestyle='solid', color='red')

        self.analysisTab._mal_ax2.plot(history.history['loss'], linestyle='dashed', color='blue')
        self.analysisTab._mal_ax2.plot(history.history['val_loss'], linestyle='dashed', color='red')

        # Create Legend
        self.analysisTab._mal_ax.legend(['Train - Accuracy', 'Test - Accuracy'], loc='upper left')
        self.analysisTab._mal_ax2.legend(['Train - Loss', 'Test - Loss'], loc='lower left')

    def display_tSNE(self, event):
        # Event is an Nx6 matrix where columns 1-3 are the spatial coordinates and columns 4-6 are the RGB values of the point
        tsne3d = event[:,:3]
        colors = event[:,3:]
        sp = gl.GLScatterPlotItem(pos=tsne3d, color=colors, size=0.25, pxMode=False)
        self.analysisTab.t_sne3d.addItem(sp)

    def set_datatype(self):
        button = self.analysisTab.data_group.checkedButton()
        if button == self.analysisTab.rms:
            if self.train_data is None:  # raise error if no data is present
                message = 'No data present, collect or load data to display plots'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                button.setChecked(False)
            elif self.rms_data is None:
                self.analysisTab.norm.setChecked(True)
                message = 'Unable to display rms data'
                QtWidgets.QMessageBox.information(self, 'Help', message)
            else:
                self.display_data(self.rms_data)
        elif button == self.analysisTab.raw:
            if self.train_data is None:  # raise error if no data is present
                message = 'No data present, collect or load data to display plots'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                button.setChecked(False)
            elif self.raw_data is None:
                self.analysisTab.norm.setChecked(True)
                message = 'Unable to display raw data'
                QtWidgets.QMessageBox.information(self, 'Help', message)
            else:
                self.display_data(self.raw_data)
        elif button == self.analysisTab.filt:
            if self.train_data is None:  # raise error if no data is present
                message = 'No data present, collect or load data to display plots'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                button.setChecked(False)
            elif self.raw_data is None:
                self.analysisTab.norm.setChecked(True)
                message = 'Unable to display filtered data'
                QtWidgets.QMessageBox.information(self, 'Help', message)
            elif self.filt_data is None:
                self.analysisTab.norm.setChecked(True)
                message = 'Unable to display filtered data'
                QtWidgets.QMessageBox.information(self, 'Help', message)
            else:
                self.display_data(self.filt_data)
        if button == self.analysisTab.norm:
            if self.train_data is None:  # raise error if no data is present
                message = 'No data present, collect or load data to display plots'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                button.setChecked(False)
            else:
                self.display_data(self.norm_data)
    
    def set_shift(self):
        data = self.train_data  # create a copy of the original data
        for key, value in self.channel_checkbox.items():
            if not value[1].isChecked() or not value[0].isChecked():
                data = data.drop(columns=[key])

        shift = int(self.analysisTab.shift.value() * self.fs) # Total number of samples to shift

        self.selected_data = data.iloc[shift:, :-1]
        self.selected_data = self.selected_data.reset_index(drop=True)
        tasks = data.iloc[:-shift, -1]
        tasks = tasks.reset_index(drop=True)
        self.selected_data = self.selected_data.assign(Task = tasks)

        self.analysisTab.down_samples.setText(f'Downsampled: {shift} samples')
        self.display_data(self.selected_data)

    def data_prediction(self, event):
        # brush = pg.mkBrush(color=(255, 0, 0))

        # indices = self.train_data[self.train_data.Task != 0]

        # t = indices.index / self.fs  # this only occurs when self.train_data is a dataframe

        task_train = event[:, 0]  # train data
        task_predict = event[:, 1]  # train data
        k = accuracy_score(task_predict, task_train)
        title = "Results using Deep Learning" "\n Accuracy: " + str(k * 100) + "%"
        # self.analysisTab.TASK.plot(t, task_predict, pen=None, symbol='o', symbolPen=None, symbolBrush=brush,
        #                            symbolSize=1)
        self.analysisTab.TASK.setTitle(title)     

    def patient_predictions(self, event):
        self.analysisTab.TASK.clear()
        brush = pg.mkBrush(color=(255, 0, 0))

        pose_predict = event
        t = np.linspace(0, len(pose_predict) / 500, num=len(pose_predict))
        title = "Testing patient using new prompt"
        self.analysisTab.TASK.plot(t, pose_predict, pen=None, symbol='o', symbolPen=None, symbolBrush=brush,
                                   symbolSize=1)
        self.analysisTab.TASK.setTitle(title)

    def save_data_file(self, event):
        time_now = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        self.data_time = time_now
        self.raw_data = event[0]
        self.filt_data = event[1]
        self.rms_data = event[2]
        self.norm_data = event[3]
        self.train_data = self.filt_data
        self.selected_data = self.train_data

        while True:
            dlg = windows.SessionData(self)
            if dlg.exec():
                if dlg.subject_name.text():
                    if dlg.set_fname.isChecked():
                        [fpath, _] = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', dlg.subject_path, "CSV files(*.csv)")
                        fpath_raw = fpath[:-4] + "_RAW.csv" # include a raw tag identifier
                        fpath_filt = fpath[:-4] + '_FILT.csv'
                        fpath_rms = fpath[:-4] + "_RMS.csv"
                    else:
                        fpath = f"{dlg.subject_path}/emg_NORM{time_now}.csv"
                        fpath_raw = f"{dlg.subject_path}/emg_RAW{time_now}.csv"
                        fpath_filt = f"{dlg.subject_path}/emg_FILT{time_now}.csv"
                        fpath_rms = f"{dlg.subject_path}/emg_RMS{time_now}.csv"
                    self.raw_data.to_csv(fpath_raw, index=False)
                    self.filt_data.to_csv(fpath_filt, index=False)
                    self.rms_data.to_csv(fpath_rms, index=False)
                    self.train_data.to_csv(fpath, index=False)

                    # Display Shift Functions
                    self.analysisTab.clear_shift.setEnabled(True)
                    self.analysisTab.set_shift.setEnabled(True)
                    self.analysisTab.layer_task.setEnabled(True)
                    self.analysisTab.shift.setEnabled(True)
                    self.analysisTab.total_samples.setText(f'Total Samples: {self.train_data.shape[0]}')

                    # Update GUI
                    message = f'Successfully collected data and saved to {fpath}!'
                    file_name = os.path.basename(fpath)
                    self.mainTab.data_status.setToolTip(str(fpath))
                    self.statusbar.showMessage(message, 3000)
                    self.mainTab.data_status.setText(file_name)
                    
                    self.analysisTab.raw.setEnabled(True)
                    self.analysisTab.filt.setEnabled(True)
                    self.analysisTab.rms.setEnabled(True)
                    self.analysisTab.norm.setEnabled(True)
                    self.analysisTab.norm.setChecked(True)  # default display is noramlized rms data
                    self.display_data(self.train_data)
                    break
                else:
                    message = 'Open or create a subject folder to save data to'
                    QtWidgets.QMessageBox.information(self, 'Help', message)
            else:
                message = 'User cancelled'
                self.statusbar.showMessage(message, 3000)
                break

    def save_data_from_menu(self, saveAs):
        if self.train_data is not None:
            if self.subject_path is not None:
                if saveAs:
                    [fpath, _] = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', self.subject_path, "CSV files(*.csv)")
                    if fpath:
                        fpath_raw = fpath[:-4] + "_RAW.csv" # include a raw tag identifier
                        fpath_rms = fpath[:-4] + "_RMS.csv"
                        if self.raw_data is not None:
                            self.raw_data.to_csv(fpath_raw, index=False)
                        if self.rms_data is not None:
                            self.rms_data.to_csv(fpath_rms, index=False)
                        self.train_data.to_csv(fpath, index=False)

                        # Update GUI
                        message = f'Successfully saved to {fpath}!'
                        file_name = os.path.basename(fpath)
                        self.mainTab.data_status.setToolTip(str(fpath))
                        self.statusbar.showMessage(message, 3000)
                        self.mainTab.data_status.setText(file_name)
                    else:
                        message = 'User cancelled'
                        self.statusbar.showMessage(message, 3000)
                else:
                    if self.mainTab.data_status.text():
                        fpath = f"{self.subject_path}/{self.mainTab.data_status.text()}.csv"
                        fpath_raw = f"{self.subject_path}/{self.mainTab.data_status.text()}_RAW.csv"
                        fpath_rms = f"{self.subject_path}/{self.mainTab.data_status.text()}_RMS.csv"
                        if self.raw_data is not None:
                            self.raw_data.to_csv(fpath_raw, index=False)
                        if self.rms_data is not None:
                            self.rms_data.to_csv(fpath_rms, index=False)
                        self.train_data.to_csv(fpath, index=False)
                    else:
                        fpath = f"{self.subject_path}/emg_NORM{self.data_time}.csv"
                        fpath_raw = f"{self.subject_path}/emg_RAW{self.data_time}.csv"
                        fpath_rms = f"{self.subject_path}/emg_RMS{self.data_time}.csv"

                    # Update GUI
                    message = f'Successfully collected data and saved to {fpath}!'
                    file_name = os.path.basename(fpath)
                    self.mainTab.data_status.setToolTip(str(fpath))
                    self.statusbar.showMessage(message, 3000)
                    self.mainTab.data_status.setText(file_name)
                
            else:
                message = 'Open or create subject folder to save data to'
                QtWidgets.QMessageBox.information(self, 'Help', message)
        else:
            message = 'Collect data to save it to file'
            QtWidgets.QMessageBox.information(self, 'Help', message)

    def save_model_from_menu(self, saveAs):
        if self.model is not None:
            if self.subject_path is not None:
                # Function in development
                pass
            else:
                message = 'Open or create subject folder to save model to'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                
        else:
            message = 'Train a model to save it'
            QtWidgets.QMessageBox.information(self, 'Help', message)

    def save_model(self, event): # update so that it saves in the subjects folder and check if the subject is already in system
        self.scaler = event[0]  # Fitted scaler
        self.model = event[1]
        
        while True:
            dlg = windows.SessionModel(self)
            if dlg.exec():
                if dlg.subject_name.text():
                    self.subject_path = dlg.subject_path
                    if dlg.model_name.text():
                        model_name = f'{dlg.model_name.text()}.keras'

                        # Check if model folder exists for subject.. if it doesn't, create new folder
                        model_dir_path = f"{self.subject_path}/Models"
                        if os.path.exists(model_dir_path):
                            model_path = f"{model_dir_path}/{model_name}"
                        else:
                            os.mkdir(model_dir_path)
                            model_path = f"{model_dir_path}/{model_name}"
                        model_path.replace("\\", "/")
                        self.model.save(model_path, overwrite=True)
                        scaler_path = f'{model_dir_path}/{dlg.model_name.text()}_scaler.save'
                        joblib.dump(self.scaler, scaler_path)

                        # Update GUI
                        self.mainTab.model_status.setToolTip(model_path)
                        message = f'{dlg.model_name.text()} successfully trained!'
                        self.statusbar.showMessage(message, 3000)
                        self.mainTab.model_status.setText(model_name)
                        break
                    else:
                        message = 'Enter model name before pressing OK'
                        QtWidgets.QMessageBox.information(self, 'Help', message)
                else:
                    message = 'Open or create a subject folder to save model to'
                    QtWidgets.QMessageBox.information(self, 'Help', message)
            else:
                message = 'User cancelled'
                self.statusbar.showMessage(message, 3000)
                break

    def time_progress(self, event):
        self.mainTab.countdown_label.setText(f"Beginning in: {event}")

    def update_img(self, event):
        if event == -1:
            self.mainTab.task_img.clear()
            self.mainTab.countdown_label.clear()
            self.pixmap = QtGui.QPixmap()
        else:
            task = event
            task_label = list(utils.pos.keys())
            self.mainTab.countdown_label.setText(task_label[task])
            self.pixmap = QtGui.QPixmap(utils.imgs[task])
            w = self.mainTab.task_img.width()
            h = self.mainTab.task_img.height()
            self.pixmap = self.pixmap.scaledToHeight(h, QtCore.Qt.TransformationMode.FastTransformation)
            self.mainTab.task_img.setPixmap(self.pixmap)

    def update_timer(self, event):
        timer = event
        self.mainTab.timer_label.setText(str(timer))

    def update_progress(self, event):
        progress = int(event * 100)  # convert to int out of 100

        self.mainTab.pb.setValue(progress)

    def update_message(self, msg):
        self.trainingDlg.message.setText(msg)

    def update_ui_on_epoch_end(self, current_epoch_number, logs):
        acc = str(np.round(logs.get('accuracy'), decimals=4))
        val_acc = str(np.round(logs.get('val_accuracy'), decimals=4))
        loss = str(np.round(logs.get('loss'), decimals=4))
        val_loss = str(np.round(logs.get('val_loss'), decimals=4))
        completed_epochs = current_epoch_number + 1
        progress = int((completed_epochs/model_utils.EPOCHS)*100)

        self.trainingDlg.acc.setText(acc)
        self.trainingDlg.val_acc.setText(val_acc)
        self.trainingDlg.loss.setText(loss)
        self.trainingDlg.val_loss.setText(val_loss)

        self.trainingDlg.training_pb.setValue(progress)
        text = f'{completed_epochs}/{model_utils.EPOCHS}'
        self.trainingDlg.epoch.setText(text)

        text = f'Completed epoch {completed_epochs} of {model_utils.EPOCHS}'
        self.trainingDlg.message.setText(text)

    def update_batch_count_on_trainingDlg_ui(self, batch_count):
        self.batch_count = batch_count

    def update_ui_on_batch_end(self, current_batch_number):
        completed_batches = current_batch_number

        text = f'{completed_batches}/{self.batch_count}'
        self.trainingDlg.batch.setText(text)

        elapsed_time = np.round(time.time() - self.start_training_time, decimals=3)
        s = np.floor(elapsed_time)
        min = int(s//60)
        s = int(s - min*60)
        ms = int(np.round(elapsed_time % 1, decimals=3) * 1000)

        text = f'Elapsed time: {min} m, {s} s, {ms} ms'
        self.trainingDlg.elapsed_time.setText(text)

    def verify_mindrove_status(self):
        try:
            self.board.prepare_session()
            self.board.start_stream()
            time.sleep(0.5)
            self.board.stop_stream()
            data = self.board.get_board_data()
            self.board.release_session()
            if data.shape[1] == 0:
                raise ValueError
            else:
                self.mainTab.mindrove_status.setText('Verified')
        except ValueError:
            id_ = 0
            msg = 'Connection timeout occured. Unable to connect to MindRove'
            event = (id_, msg)
            self.emit_error_message(event)
        except MindRoveError:
            id_ = 0
            msg = 'Connection timeout occured. Unable to connect to MindRove'
            event = (id_, msg)
            self.emit_error_message(event)

    def begin_collecting_data(self):
        # Create QThread objects
        self.collectThread = QtCore.QThread()
        self.countdownThread = QtCore.QThread()

        # Create worker objects
        self.collectWorker = CollectWorker(self.tasks, self.train_data, self.train_reps, self.board, self.scaler, self.active_time, self.inactive_time)
        self.countdownWorker = CountdownWorker(5)  # countdown of five

        # self.timerWorker = TimerWorker(self.active_time, self.inactive_time, self.tasks)
        # self.timerThread = QtCore.QThread()

        # Move workers to the threads
        self.collectWorker.moveToThread(self.collectThread)
        self.countdownWorker.moveToThread(self.countdownThread)
        # self.timerWorker.moveToThread(self.timerThread)
        # Connect signals and slots
        self.collectThread.started.connect(self.collectWorker.run)
        self.collectWorker.finished.connect(self.collectThread.quit)
        self.collectWorker.finished.connect(self.collectWorker.deleteLater)
        self.collectThread.finished.connect(self.collectThread.deleteLater)
        self.collectWorker.data.connect(self.save_data_file)
        self.collectWorker.updateImg.connect(self.update_img)
        self.collectWorker.updateProgress.connect(self.update_progress)
        self.collectWorker.error.connect(self.emit_error_message)

        self.countdownThread.started.connect(self.countdownWorker.run)
        self.countdownWorker.finished.connect(self.collectThread.start)
        # self.countdownWorker.finished.connect(self.timerThread.start)
        self.countdownWorker.finished.connect(self.countdownThread.quit)
        self.countdownWorker.finished.connect(self.countdownWorker.deleteLater)
        self.countdownThread.finished.connect(self.countdownThread.deleteLater)
        self.countdownThread.finished.connect(
            lambda: self.mainTab.countdown_label.clear()
        )
        self.countdownThread.finished.connect(
            lambda: self.mainTab.pb.setVisible(True)
        )
        self.countdownThread.finished.connect(
            lambda: self.mainTab.title_label.setHidden(True)
        )
        self.countdownWorker.updateTime.connect(self.time_progress)

        # Link timer thread with the collect thread via the collectWorker
        # self.timerThread.started.connect(self.timerWorker.run)
        # self.timerWorker.finished.connect(self.timerThread.quit)
        # self.timerWorker.finished.connect(self.timerWorker.deleteLater)
        # self.timerWorker.finished.connect(self.timerThread.deleteLater)
        # self.timerWorker.finished.connect(
        #     lambda: self.mainTab.timer_label.clear()
        # )
        
        # self.timerWorker.updateTimer.connect(self.update_timer)
        # Start the thread
        if len(self.tasks) > 0:
            self.countdownThread.start()
            self.mainTab.collect.setEnabled(False)
            self.mainTab.train.setEnabled(False)
            self.mainTab.test.setEnabled(False)
        else:
            message = 'Insert a prompt, found under "Tools", to begin collecting data'
            QtWidgets.QMessageBox.information(self, 'Help', message)

        # Reset tasks
        self.collectThread.finished.connect(
            lambda: self.mainTab.collect.setEnabled(True)
        )
        self.collectThread.finished.connect(
            lambda: self.mainTab.train.setEnabled(True)
        )
        self.collectThread.finished.connect(
            lambda: self.mainTab.test.setEnabled(True)
        )
        self.collectThread.finished.connect(
            lambda: self.mainTab.pb.setHidden(True)
        )
        self.collectThread.finished.connect(
            lambda: self.mainTab.pb.setValue(0)
        )
        self.collectThread.finished.connect(
            lambda: self.mainTab.title_label.setVisible(True)
        )
        # self.collectThread.finished.connect(
        #     lambda: self.timerWorker.interrupt()
        # )

    def begin_training_data(self):
        # Create QThread objects
        self.trainThread = QtCore.QThread()

        # Create worker objects
        self.trainWorker = TrainWorker(self.selected_data, self.model, self.scaler)  # specify the channels to use to train

        # Move workers to the threads
        self.trainWorker.moveToThread(self.trainThread)

        # Connect signals and slots
        self.trainThread.started.connect(self.trainWorker.run)
        self.trainWorker.finished.connect(self.trainThread.quit)
        self.trainWorker.finished.connect(self.trainWorker.deleteLater)
        self.trainThread.finished.connect(self.trainThread.deleteLater)

        self.trainWorker.paramSignal.connect(self.save_model)
        self.trainWorker.dataSignal.connect(self.data_prediction)
        self.trainWorker.histSignal.connect(self.display_model_history)
        self.trainWorker.tsneSignal.connect(self.display_tSNE)
        self.trainWorker.cmSignal.connect(self.display_confusion_matrix)

        self.trainWorker.batch_count_signal.connect(self.update_batch_count_on_trainingDlg_ui)
        self.trainWorker.epoch_end_signal.connect(self.update_ui_on_epoch_end)
        self.trainWorker.batch_end_signal.connect(self.update_ui_on_batch_end)
        self.trainWorker.emit_message_signal.connect(self.update_message)

        # Start the thread

        if self.train_data is None:
            message = 'Unable to train model with no data'
            QtWidgets.QMessageBox.warning(self, 'Help', message)
        else:
            self.mainTab.collect.setEnabled(False)
            self.mainTab.train.setEnabled(False)
            self.mainTab.test.setEnabled(False)
            self.start_training_time = time.time()
            self.trainThread.start()
            self.trainingDlg = windows.TrainingDialog(self)
            self.trainingDlg.message.setText('Performing Feature Extraction')

            # Reset tasks
            self.trainThread.finished.connect(
                lambda: self.mainTab.collect.setEnabled(True)
            )
            self.trainThread.finished.connect(
                lambda: self.mainTab.train.setEnabled(True)
            )
            self.trainThread.finished.connect(
                lambda: self.mainTab.test.setEnabled(True)
            )
            self.trainThread.finished.connect(
                lambda: self.trainingDlg.close()
            )

    def begin_testing_data(self):
        # Create QThread objects
        self.testThread = QtCore.QThread()
        self.countdownThread = QtCore.QThread()

        if self.serial_obj is not None:
            # Create worker objects
            # Must use the selected channels after training | add some sort of flag if loading a model without training further
            if self.selected_data is not None:
                selected_data = list(self.selected_data)
                if 'Task' in selected_data:
                    selected_data.remove('Task')
            else:
                selected_data = self.selected_data
            
            dlg = windows.TestingDialog(self)

            if dlg.exec():
                active_time = dlg.active_time.value()
                inactive_time = dlg.inactive_time.value()
                fixed_mode = dlg.fixed_mode.isChecked()

                self.testWorker = TestWorker(self.tasks, selected_data, self.train_reps,
                                            self.model, self.scaler, self.board, self.serial_obj,
                                            active_time, inactive_time, fixed_mode=fixed_mode)
                self.countdownWorker = CountdownWorker(5)
                # Move workers to the threads
                self.testWorker.moveToThread(self.testThread)
                self.countdownWorker.moveToThread(self.countdownThread)
                # Connect signals and slots
                self.testThread.started.connect(self.testWorker.run)
                self.testWorker.finished.connect(self.testThread.quit)
                self.testWorker.finished.connect(self.testWorker.deleteLater)
                self.testThread.finished.connect(self.testThread.deleteLater)
                self.testWorker.updateImg.connect(self.update_img)
                self.testWorker.predictions.connect(self.patient_predictions)
                self.testWorker.error.connect(self.emit_error_message)

                self.countdownThread.started.connect(self.countdownWorker.run)
                self.countdownWorker.finished.connect(self.testThread.start)
                self.countdownWorker.finished.connect(self.countdownThread.quit)
                self.countdownWorker.finished.connect(self.countdownWorker.deleteLater)
                self.countdownThread.finished.connect(self.countdownThread.deleteLater)
                self.countdownThread.finished.connect(
                    lambda: self.mainTab.countdown_label.clear()
                )
                self.countdownThread.finished.connect(
                    lambda: self.mainTab.title_label.setHidden(True)
                )

                self.countdownWorker.updateTime.connect(self.time_progress)

                # Start the thread
                if self.model is None:
                    message = 'Cannot test a non-existent model'
                    QtWidgets.QMessageBox.warning(self, 'Help', message)

                elif len(self.tasks) == 0:
                    message = 'Insert a task sequence, found under "Tools", to test model'
                    QtWidgets.QMessageBox.warning(self, 'Help', message)
                else:
                    self.countdownThread.start()
                    self.mainTab.collect.setEnabled(False)
                    self.mainTab.train.setEnabled(False)
                    self.mainTab.test.setEnabled(False)

                    # Reset tasks
                    self.testThread.finished.connect(
                        lambda: self.mainTab.collect.setEnabled(True)
                    )
                    self.testThread.finished.connect(
                        lambda: self.mainTab.train.setEnabled(True)
                    )
                    self.testThread.finished.connect(
                        lambda: self.mainTab.test.setEnabled(True)
                    )
                    self.testThread.finished.connect(
                        lambda: self.mainTab.detection_label.clear()
                    )
                    self.testThread.finished.connect(
                        lambda: self.mainTab.title_label.setVisible(True)
                    )
        else:
            message = 'Port is not connected'
            QtWidgets.QMessageBox.warning(self, 'Warning', message)


# Press the play button in the gutter to run the script.
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = PoseApp()
    sys.exit(app.exec())
