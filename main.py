import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import pyqtgraph as pg
import serial
from PyQt6 import QtWidgets, QtCore, QtGui, QtSerialPort
from keras.layers import Dense
from keras.models import Sequential
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import FilterTypes, DataFilter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.optimizers import Adam
import joblib

import emg_classification
import settings
from windows import createMain, createAnalysis, createPatientView, CheckBoxDialog, FeedbackWin


# Some icons by Yusuke Kamiyamane. Licensed under a Creative Commons Attribution 3.0 License.


def createBoard():
    params = MindRoveInputParams()
    board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD.value, params)
    return board


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

    def __init__(self, time_input, train_reps):
        super().__init__()
        self.timer = time_input
        self.reps = train_reps

    def run(self):
        for i in range(self.reps):
            timer = self.timer
            while timer:
                self.updateTimer.emit(timer)
                time.sleep(1)
                timer -= 1
        self.finished.emit()


class CollectWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateImg = QtCore.pyqtSignal()
    data = QtCore.pyqtSignal(tuple)
    error = QtCore.pyqtSignal()
    updateProgress = QtCore.pyqtSignal(float)

    def __init__(self, prompt, train_data, train_reps, board, collect_time):
        super().__init__()

        self.prompt = prompt
        self.train_data = train_data
        self.train_reps = train_reps
        self.board = board
        self.collect_time = collect_time
        self.fs = BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD.value)
        self.filter_type = FilterTypes.BUTTERWORTH
        self.ws = 50

    def stream(self):
        try:
            emg_channels = self.board.get_emg_channels(BoardIds.MINDROVE_WIFI_BOARD.value)
            data_w_pose = np.ndarray(shape=(len(emg_channels) + 1, 0), dtype=np.float64)
            for i in range(self.train_reps):
                self.updateImg.emit()
                k = int(self.prompt[i])
                self.board.start_stream()
                time.sleep(self.collect_time)
                self.board.stop_stream()
                data = self.board.get_board_data()

                # Processing Data
                data[8, :] = k
                data_w_pose = np.concatenate((data_w_pose, data[0:9, :]), axis=1)
                progress = (i + 1) / self.train_reps
                self.updateProgress.emit(progress)
            self.board.release_session()

            dataDF = pd.DataFrame(np.transpose(data_w_pose),
                                  columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])
            dataDF = dataDF.reset_index(drop=True)
            data_filt = data_w_pose.copy()

            for count, channel in enumerate(emg_channels):
                DataFilter.perform_bandpass(data_filt[channel], self.fs, 125, 240, 4, self.filter_type, 0)

            for count, channel in enumerate(emg_channels):
                DataFilter.perform_lowpass(data_filt[channel], self.fs, 250, 4, self.filter_type, 0)

            filtered = pd.DataFrame(np.transpose(data_filt[0:8, :]),
                                    columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])
            smoothed = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])

            filtered_trimmed = filtered.iloc[500:, :]
            data_trimmed = dataDF.iloc[500:, :]
            data_trimmed = data_trimmed.reset_index(drop=True)

            for i in filtered_trimmed.columns:
                smoothed[i] = emg_classification.window_rms(filtered_trimmed[i], self.ws)

            filtered_trimmed = filtered_trimmed.assign(Pose=data_trimmed.Pose)
            smoothed.Pose = data_trimmed.Pose

            data = (filtered_trimmed, smoothed)  # RAW [0] ... RMS [1]

            self.updateImg.emit()
            self.finished.emit()
            self.data.emit(data)
        except BrokenPipeError:
            if self.board.is_prepared():
                self.board.release_session()
            self.error.emit()

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

    def __init__(self, train_data, model, scaler):
        super().__init__()
        self.train_data = train_data
        self.model = model
        self.scaler = scaler

    def run(self):
        # From sklearn.metrics import classification_report
        X = self.train_data.iloc[:, :len(self.train_data.columns) - 1]  # get channels while excluding pose column

        # int value of the number of unique classes being trained
        classes = int(self.train_data.max(axis=0, skipna=True, numeric_only=True).iloc[-1]) + 1

        self.scaler = StandardScaler()
        y = emg_classification.get_one_hot(self.train_data.Pose.astype(int), classes)
        scaled_channels = self.scaler.fit_transform(X)

        # 250 ms frame shift
        scaled_channels_trim = scaled_channels[125:, :]
        y_trim = y[:len(scaled_channels_trim), :]

        if self.model is None:
            self.model = Sequential()
            self.model.add(Dense(256, activation='relu', input_shape=(len(self.train_data.columns) - 1,)))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(classes, activation="softmax"))

            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(learning_rate=0.01),
                               metrics=['accuracy'])
        self.model.fit(scaled_channels_trim, y_trim, epochs=50, batch_size=100, shuffle=False, verbose=1, validation_split=0.3)

        y_predicted = self.model.predict(x=scaled_channels_trim, batch_size=100, verbose=0)

        predicted_labels = np.argmax(y_predicted, axis=-1)  # array of predicted labels
        true_labels = np.argmax(y_trim, axis=-1)  # array of true labels

        data = np.column_stack((true_labels, predicted_labels))  # first column [0]->train | second column [1]->predict

        parameters = (self.scaler, self.model)

        self.paramSignal.emit(parameters)  # tuple containing the fitted scaler and model
        self.dataSignal.emit(data)  # numpy 2-D array of poses
        self.finished.emit()


class TestWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateImg = QtCore.pyqtSignal()
    validatePose = QtCore.pyqtSignal(list)
    predictions = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, prompt, selected_channels, train_reps, model, scaler, board, serial_obj, collect_time):
        super().__init__()

        self.prompt = prompt
        self.channels = selected_channels  # The selected channels feed into the model
        self.train_reps = train_reps
        self.model = model
        self.scaler = scaler
        self.board = board
        self.serial_obj = serial_obj
        self.collect_time = collect_time
        # Constants
        self.fs = BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD.value)
        self.filter_type = FilterTypes.BUTTERWORTH
        self.ws = 50

    def test(self):
        test_data = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])
        for i in test_data.columns:
            if i not in self.channels:
                test_data.drop(columns=[i])

        emg_channels = self.board.get_emg_channels(BoardIds.MINDROVE_WIFI_BOARD.value)
        poses = list()

        predicted_labels = np.ndarray(shape=(0,), dtype=np.int64)

        for i in range(self.train_reps):
            start = time.time()
            self.updateImg.emit()
            k = int(self.prompt[i])
            self.board.start_stream()
            time.sleep(self.collect_time)
            self.board.stop_stream()
            data = self.board.get_board_data()
            # Processing Data
            data_filt = data[0:8, :]

            for count, channel in enumerate(emg_channels):
                DataFilter.perform_bandpass(data_filt[channel], self.fs, 125, 240, 4, self.filter_type, 0)

            for count, channel in enumerate(emg_channels):
                DataFilter.perform_lowpass(data_filt[channel], self.fs, 250, 4, self.filter_type, 0)

            filtered = pd.DataFrame(np.transpose(data_filt[0:8, :]),
                                    columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])
            smoothed = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])

            filtered_trimmed = filtered.iloc[1000:, :]  # trim the first and last second of contraction

            for j in filtered_trimmed.columns:
                smoothed[j] = emg_classification.window_rms(filtered_trimmed[j], self.ws)

            for j in smoothed.columns:  # Check which columns are not included, then drop...
                if j not in self.channels:
                    smoothed.drop(columns=[j])
            # should find a more efficient way going forward ^

            test_data = pd.concat([test_data, smoothed])

            X = self.scaler.transform(smoothed)

            # 250 ms frame shift
            X_scaled_trimmed = X[125:, :]
            print(X_scaled_trimmed.shape)
            Y = self.model.predict(X_scaled_trimmed)
            print(Y.shape)

            labels = np.argmax(Y, axis=-1)  # array of predicted labels
            predicted_labels = np.concatenate([predicted_labels, labels], dtype=np.int64)

            unique, indices, counts = np.unique(labels, return_counts=True, return_index=True)
            predicted_label = np.argmax(counts)
            print(unique, indices, counts)
            print(predicted_label)

            validate = [k, predicted_label]  # [intended, detected]
            self.validatePose.emit(validate)

            value = settings.pos_list[predicted_label]
            self.serial_obj.write(value)
            time.sleep(self.collect_time)

            poses.append(predicted_label)

            end = time.time()
            print(end-start)

        print(poses)
        f = 'C:/Users/mppen/OneDrive/Documents/UMD/NueroLab/Pose Classification/Model_optimization/mppen_testdata.csv'
        test_data.to_csv(f, index=False)
        self.updateImg.emit()
        self.predictions.emit(predicted_labels)
        self.board.release_session()
        self.finished.emit()

    def run(self):
        if self.board.is_prepared():
            self.test()
        else:
            self.board.prepare_session()
            self.test()


class PoseApp(QtWidgets.QMainWindow):
    train_data = np.ndarray(shape=(0, 9), dtype=np.int64)
    td_selchan = None  # specify the selected channels

    def __init__(self):
        # Call the Parent constructor
        super().__init__()

        self.raw_data = None
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

        self.count = -1
        self.prompt = []
        self.train_reps = len(self.prompt)
        self.collect_time = 4  # Default
        self.timer = self.collect_time
        self.test_data = None
        self.scaler = None
        self.model = None
        self.table = None
        self.serial_obj = None
        self.board = createBoard()
        self.fs = BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD.value)
        self.ports = None

        self.setWindowTitle("EMG Intention Decoding")
        self.setGeometry(200, 100, 1200, 875)

        tabs = QtWidgets.QTabWidget(self)
        self.mainTab = createMain()
        tabs.addTab(self.mainTab, "Main")
        self.analysisTab = createAnalysis()
        tabs.addTab(self.analysisTab, "Analysis")

        self.setCentralWidget(tabs)

        self._createActions()
        self._createMenuBar()
        self._createStatusBar()
        self._connectActions()

        #  self.mindRoveStatus()

        self.mainTab.collect.pressed.connect(self.collectData)
        self.mainTab.train.pressed.connect(self.trainData)
        self.mainTab.test.pressed.connect(self.testData)
        self.mainTab.detect.clicked.connect(self.detectAvailablePorts)
        self.mainTab.connect.clicked.connect(self.connect_func)

        self.channels = {'CH0': self.analysisTab.CH0,
                         'CH1': self.analysisTab.CH1,
                         'CH2': self.analysisTab.CH2,
                         'CH3': self.analysisTab.CH3,
                         'CH4': self.analysisTab.CH4,
                         'CH5': self.analysisTab.CH5,
                         'CH6': self.analysisTab.CH6,
                         'CH7': self.analysisTab.CH7,
                         'Pose': self.analysisTab.POSE}

        self.channel_checkbox = {'CH0': self.analysisTab.ch0_check,
                                 'CH1': self.analysisTab.ch1_check,
                                 'CH2': self.analysisTab.ch2_check,
                                 'CH3': self.analysisTab.ch3_check,
                                 'CH4': self.analysisTab.ch4_check,
                                 'CH5': self.analysisTab.ch5_check,
                                 'CH6': self.analysisTab.ch6_check,
                                 'CH7': self.analysisTab.ch7_check}

        self.channel_main_ui = {'CH0': self.mainTab.ch0_main,
                                'CH1': self.mainTab.ch1_main,
                                'CH2': self.mainTab.ch2_main,
                                'CH3': self.mainTab.ch3_main,
                                'CH4': self.mainTab.ch4_main,
                                'CH5': self.mainTab.ch5_main,
                                'CH6': self.mainTab.ch6_main,
                                'CH7': self.mainTab.ch7_main}

        self.analysisTab.ch0_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch0_check))
        self.analysisTab.ch1_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch1_check))
        self.analysisTab.ch2_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch2_check))
        self.analysisTab.ch3_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch3_check))
        self.analysisTab.ch4_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch4_check))
        self.analysisTab.ch5_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch5_check))
        self.analysisTab.ch6_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch6_check))
        self.analysisTab.ch7_check.clicked.connect(lambda: self.updatePlot(self.analysisTab.ch7_check))

        self.analysisTab.display_all.clicked.connect(self.displayAllChannels)
        self.analysisTab.clear_all.clicked.connect(self.clearChannelPlots)
        self.analysisTab.set_channels.clicked.connect(self.setChannels)

        self.analysisTab.data_group.buttonClicked.connect(self.setRawRms)

        self.patientUI = createPatientView()
        self.patientUI.show()

        self.show()

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = menuBar.addMenu("File")
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.loadDataAction)
        fileMenu.addAction(self.loadModelAction)
        # Clear menu
        clearMenu = fileMenu.addMenu("Clear")
        clearMenu.addAction(self.clearDataAction)
        clearMenu.addAction(self.clearModelAction)
        clearMenu.addAction(self.clearAllAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Tools menu
        toolMenu = menuBar.addMenu("Tools")
        promptMenu = toolMenu.addMenu("Prompt")
        promptMenu.addAction(self.newPromptAction)
        promptMenu.addAction(self.randPromptAction)
        promptMenu.addAction(self.randLenPromptAction)
        toolMenu.addAction(self.createTrainingAction)
        toolMenu.addAction(self.changeTimeAction)

        # Help menu
        helpMenu = menuBar.addMenu("Help")
        helpMenu.addAction(self.documentationAction)
        helpMenu.addAction(self.aboutAction)
        helpMenu.addAction(self.feedbackAction)

    def _createActions(self):
        # Creating actions using the second constructor
        self.loadDataAction = QtGui.QAction("Open &Data...", self)
        self.loadModelAction = QtGui.QAction("Open &Model...", self)
        self.clearDataAction = QtGui.QAction("Clear Data", self)
        self.clearModelAction = QtGui.QAction("Clear Model", self)
        self.clearAllAction = QtGui.QAction("Clear All", self)
        self.exitAction = QtGui.QAction("&Exit", self)
        self.documentationAction = QtGui.QAction("&Documentation", self)
        self.aboutAction = QtGui.QAction("&About", self)
        self.feedbackAction = QtGui.QAction('Send &Feedback', self)
        self.newPromptAction = QtGui.QAction("&New Prompt...", self)
        self.randPromptAction = QtGui.QAction("&Random Prompt", self)
        self.randLenPromptAction = QtGui.QAction("Random &Length Prompt", self)
        self.createTrainingAction = QtGui.QAction("Training Parameters", self)
        self.changeTimeAction = QtGui.QAction("Set Time Interval", self)
        # self.connectPortAction = QtGui.QAction("Connect Port...", self)

    def _connectActions(self):
        # Connect File actions
        self.loadDataAction.triggered.connect(self.loadData)
        self.loadModelAction.triggered.connect(self.loadModel)
        self.clearDataAction.triggered.connect(self.clearData)
        self.clearModelAction.triggered.connect(self.clearModel)
        self.clearAllAction.triggered.connect(self.clearAll)
        self.exitAction.triggered.connect(self.close)

        # Connect Tools actions
        self.newPromptAction.triggered.connect(self.newPrompt)
        self.randPromptAction.triggered.connect(self.randPrompt)
        self.randLenPromptAction.triggered.connect(self.randLenPrompt)
        self.createTrainingAction.triggered.connect(self.trainingParameters)
        self.changeTimeAction.triggered.connect(self.changeTime)

        # Connect Icon actions
        self.mainTab.prompt_icon.clicked.connect(self.newPrompt)
        self.mainTab.data_icon.clicked.connect(self.loadData)
        self.mainTab.model_icon.clicked.connect(self.loadModel)
        #  self.mainTab.port_icon.clicked.connect(self.newPrompt)

        # Connect Help actions
        self.documentationAction.triggered.connect(emg_classification.documentation)
        self.aboutAction.triggered.connect(self.about)
        self.feedbackAction.triggered.connect(self.send_feedback)


    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        # Adding a temporary message
        self.statusbar.showMessage("Ready", 3000)
        self.connection_status = QtWidgets.QLabel(self.current_port_status())
        self.statusbar.addPermanentWidget(self.connection_status)

    def detectAvailablePorts(self):
        self.mainTab.port_select.clear()
        self.ports = QtSerialPort.QSerialPortInfo().availablePorts()

        if len(self.ports) != 0:
            self.mainTab.port_select.clear()
            for port in self.ports:
                self.mainTab.port_select.addItem(port.portName())
        else:
            self.mainTab.port_select.addItem('Not Available')
            self.mainTab.port_status.setText('Not Connected')

    def current_port_status(self):
        if self.serial_obj is None:
            x = 'Device not connected'
        else:
            x = f'Device is connected to {self.serial_obj.port}'
        return x

    #  function to connect to microcontroller... configure so that baudrate match and are optimized
    def connect_func(self):
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

    def trainingParameters(self):
        dlg = CheckBoxDialog(self)

        if dlg.exec():
            self.prompt = []
            display = []
            for key, value in dlg.poses.items():
                prompt = []
                if key.isChecked():
                    prompt.append(value)
                    prompt.append(0)
                    for i in range(4):  # increase prompt length by 2^5 for 16 iterations of the prompt
                        prompt = prompt + prompt
                    self.prompt = self.prompt + prompt
                    display.append(value)
            self.prompt = [0] + self.prompt

            self.mainTab.prompt_status.setText(f'{display}')
            self.train_reps = len(self.prompt)
        else:
            message = 'User cancelled'
            self.statusbar.showMessage(message)

    def changeTime(self):
        new_interval, done1 = QtWidgets.QInputDialog.getInt(
            self, 'Enter new time interval:', f'Current interval: {self.collect_time}', 1, 2, 10, 1)

        if done1:
            self.collect_time = new_interval

    def setChannels(self):
        self.td_selchan = self.train_data  # create a copy of the original data
        for key, value in self.channel_checkbox.items():
            if not value.isChecked():
                self.td_selchan = self.td_selchan.drop(columns=[key])

    def configChannels(self):  # function if loading a model directly

        for count, channel_name in enumerate(self.channel_checkbox):
            channel_main = self.channel_main_ui.get(channel_name)
            channel_main.setStyleSheet("font-weight: bold; color: black")
            channel_check = self.channel_checkbox.get(channel_name)
            channel_check.setEnabled(True)
            channel_check.setChecked(True)

        self.td_selchan = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])

    def loadData(self):
        try:
            filt = "CSV files(*.csv);;XML files(*.xml);;Text files (*.txt)"
            fname, selFilter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', filt)
            self.train_data = pd.read_csv(fname)
            message = f'{fname} successfully uploaded!'
            file_name = os.path.basename(fname)
            self.statusbar.showMessage(message)
            self.mainTab.data_status.setText(file_name)
            self.analysisTab.raw.setEnabled(True)
            self.analysisTab.rect.setEnabled(True)
            self.analysisTab.rect.setChecked(True)  # default display is rectified data
            self.updateData(self.train_data)
            self.setChannels()
        except FileNotFoundError:
            message = 'User Cancelled'
            self.statusbar.showMessage(message)

    def loadModel(self):
        folder_path = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Model"))
        try:
            model_path = f'{folder_path}/model'
            self.model = keras.models.load_model(model_path)
            scaler_path = f'{folder_path}/scaler.save'
            self.scaler = joblib.load(scaler_path)
            message = f'{folder_path} successfully uploaded!'
            folder_name = os.path.basename(folder_path)
            self.statusbar.showMessage(message)
            self.mainTab.model_status.setText(folder_name)
            self.configChannels()
        except OSError:
            message = 'User Cancelled'
            self.statusbar.showMessage(message)

    def clearData(self):
        self.train_data = np.ndarray(shape=(0, 9), dtype=np.int64)
        self.mainTab.data_status.clear()
        self.analysisTab.table.setModel(None)
        self.clearPlots()

        for key, value in self.channel_checkbox.items():
            value.setEnabled(False)

        self.analysisTab.raw.setEnabled(False)
        self.analysisTab.rect.setEnabled(False)

    def clearPlots(self):
        for key, value in self.channel_main_ui.items():
            value.setStyleSheet("font-weight: bold; color: gray")

        for key, value in self.channel_checkbox.items():
            value.setChecked(False)

        plots = self.channels.values()
        for plot_item in plots:
            plot_item.clear()

    def clearChannelPlots(self):
        for key, value in self.channel_main_ui.items():
            value.setStyleSheet("font-weight: bold; color: gray")

        for key, value in self.channel_checkbox.items():
            value.setChecked(False)

        plots = self.channels.values()
        for plot_item in plots:
            if plot_item != self.analysisTab.POSE:
                plot_item.clear()

    def clearModel(self):
        self.model = None
        self.mainTab.model_status.clear()

    def clearAll(self):
        self.clearData()
        self.clearModel()

    def newPrompt(self):
        # Generate a new prompt with any Pose assignment between 0 and 20

        while True:
            new_prompt, done1 = QtWidgets.QInputDialog.getText(
                self, 'New Prompt', 'Enter new Prompt:')
            new_prompt = new_prompt.split()
            for i in range(len(new_prompt)):
                if new_prompt[i].isdigit():
                    new_prompt[i] = int(new_prompt[i])

            if all([isinstance(item, int) for item in new_prompt]):
                if all([int(item) >= 0 for item in new_prompt]) and all([int(item) <= 4 for item in new_prompt]):
                    if len(new_prompt) <= 30:
                        if len(new_prompt) > 0 and new_prompt[0] != 0:
                            new_prompt = [0] + new_prompt  # add a rest pose at beginning to normalize data processing
                        self.prompt = new_prompt
                        self.mainTab.prompt_status.setText(f'{self.prompt}')
                        self.train_reps = len(self.prompt)
                        break
                    else:
                        message = 'Prompt length should be no more than 30'
                        QtWidgets.QMessageBox.information(self, 'Help', message)
                        continue
                else:
                    message = 'Integers should be between 0 and 4'
                    QtWidgets.QMessageBox.information(self, 'Help', message)
                    continue
            else:
                message = 'Create input as integers separated by space'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                continue

    def randPrompt(self):
        # Generate a randomized prompt of any length between 0 and 20
        message = 'Enter the number of poses you want to implement (maximum of 20 allowed):'
        length, done1 = QtWidgets.QInputDialog.getInt(self, 'Assign a Random Prompt', message, min=0, max=20)

        new_prompt = np.random.randint(0, high=5, size=(1, int(length)))
        new_prompt = new_prompt.tolist()  # converts into a list within a list
        new_prompt = new_prompt[0]  # obtain the list
        if new_prompt[0] != 0:
            new_prompt = [0] + new_prompt  # add a rest pose at beginning to normalize data processing
        self.prompt = new_prompt  # assign local var to global
        self.train_reps = len(self.prompt)
        self.mainTab.prompt_status.setText(f'{self.prompt}')

    def randLenPrompt(self):
        # Generate a pseudorandom prompt up to 20 poses in length
        length = np.random.randint(4, high=21)
        new_prompt = np.random.randint(0, high=5, size=(1, length))
        new_prompt = new_prompt.tolist()
        new_prompt = new_prompt[0]  # obtain the list
        if new_prompt[0] != 0:
            new_prompt = [0] + new_prompt  # add a rest pose at beginning to normalize data processing
        self.prompt = new_prompt  # assign local var to global
        self.train_reps = len(self.prompt)
        self.mainTab.prompt_status.setText(f'{self.prompt}')

    def about(self):
        # Logic for showing an 'about' dialog content goes here...
        message = "Collect EMG data from a patient and train a model to detect the intention behind a patient's actions"
        QtWidgets.QMessageBox.information(self, 'EMG intention detection', message)

    def send_feedback(self):
        from email.message import EmailMessage
        import smtplib
        import ssl
        import os
        dlg = FeedbackWin(self)

        if dlg.exec():
            name = dlg.name.text()

            feedback = dlg.feedback.toPlainText()

            # replace with
            password = os.environ.get('ID_EMAIL_PASSWORD')

            port = 465  # For SSL
            smtp_server = "smtp.gmail.com"
            email_sender = os.environ.get('ID_EMAIL_USERNAME')
            email_receiver = os.environ.get('ID_EMAIL_USERNAME')

            subject = f'Feedback from {name}'
            body = feedback

            em = EmailMessage()
            em['From'] = email_sender
            em['To'] = email_receiver
            em['Subject'] = subject
            em.set_content(body)

            context = ssl.create_default_context()

            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(email_sender, password)
                server.sendmail(email_sender, email_receiver, em.as_string())

        else:
            message = 'User cancelled'
            self.statusbar.showMessage(message)

    def closeEvent(self, event):
        if self.board.is_prepared():
            close = QtWidgets.QMessageBox.question(self,
                                                   "QUIT",
                                                   "ARE YOU SURE? Quitting will interrupt data collection",
                                                   QtWidgets.QMessageBox.StandardButton.Yes |
                                                   QtWidgets.QMessageBox.StandardButton.No)
            if close == QtWidgets.QMessageBox.StandardButton.Yes:  # add boolean statement for threads
                if self.serial_obj is not None and self.serial_obj.is_open:
                    self.serial_obj.close()
                self.board.release_session()
                self.patientUI.close()
                event.accept()
            elif close == QtWidgets.QMessageBox.StandardButton.No:
                event.ignore()
        else:
            if self.serial_obj is not None and self.serial_obj.is_open:
                self.serial_obj.close()
            self.patientUI.close()
            event.accept()

    def saveDataFile(self, event):
        time_now = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        self.raw_data = event[0]
        self.train_data = event[1]
        self.td_selchan = self.train_data

        filename = QtWidgets.QMessageBox.question(QtWidgets.QWidget(), "Save Data", "Create new file name.",
                                                  QtWidgets.QMessageBox.StandardButton.Yes |
                                                  QtWidgets.QMessageBox.StandardButton.No)
        home_dir = str(os.path.expanduser('~'))
        fpath = str
        if filename == QtWidgets.QMessageBox.StandardButton.Yes:
            [fpath, _] = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', home_dir, "CSV files(*.csv)")
        elif filename == QtWidgets.QMessageBox.StandardButton.No:
            fpath = f"{home_dir}/emg_data{time_now}.csv"
        self.train_data.to_csv(fpath, index=False)
        message = f'Successfully collected data and saved to {fpath}!'
        file_name = os.path.basename(fpath)
        self.statusbar.showMessage(message)
        self.mainTab.data_status.setText(file_name)
        self.analysisTab.raw.setEnabled(True)
        self.analysisTab.rect.setEnabled(True)
        self.analysisTab.rect.setChecked(True)  # default display is rectified data
        self.updateData(self.train_data)

    def updateData(self, data):
        self.clearPlots()
        t2 = data.index / self.fs

        for count, channel_name in enumerate(data):
            channel = self.channels.get(channel_name)  # gets the plot
            channel.plot(t2, data[channel_name].to_numpy())

            if channel_name != 'Pose':  # ignore the channel containing pose
                channel_main = self.channel_main_ui.get(channel_name)
                channel_main.setStyleSheet("font-weight: bold; color: black")
                channel_check = self.channel_checkbox.get(channel_name)
                channel_check.setEnabled(True)
                channel_check.setChecked(True)

        self.table = emg_classification.TableModel(data)
        self.analysisTab.table.setModel(self.table)

    def updatePlot(self, checkbox):
        if len(self.train_data) > 0 or type(self.train_data) is pd.DataFrame:
            channel = self.channels.get(checkbox.text())  # get the plot object from the name of the checkbox
            channel_ui = self.channel_main_ui.get(checkbox.text())  # get the channel interface object on the main tab

            if checkbox.isChecked():
                t2 = self.train_data.index / self.fs
                channel.plot(t2, self.train_data[checkbox.text()].to_numpy())  # plot the data one specified plot
                channel_ui.setStyleSheet("font-weight: bold; color: black")
            else:
                channel.clear()  # clear the data from the specified plot
                channel_ui.setStyleSheet("font-weight: bold; color: gray")

    def displayAllChannels(self):
        for i in self.channel_checkbox.values():
            i.setChecked(True)
            self.updatePlot(i)

    def setRawRms(self):
        button = self.analysisTab.data_group.checkedButton()
        if button == self.analysisTab.rect:
            if len(self.train_data) == 0:  # raise error if no data is present
                message = 'No data present, collect or load data to display plots'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                button.setChecked(False)
            else:
                self.updateData(self.train_data)
        elif button == self.analysisTab.raw:
            if len(self.train_data) == 0:  # raise error if no data is present
                message = 'No data present, collect or load data to display plots'
                QtWidgets.QMessageBox.information(self, 'Help', message)
                button.setChecked(False)
            elif self.raw_data is None:
                self.analysisTab.rect.setChecked(True)
                message = 'Unable to display raw data'
                QtWidgets.QMessageBox.information(self, 'Help', message)
            else:
                self.updateData(self.raw_data)

    def dataPrediction(self, event):
        brush = pg.mkBrush(color=(255, 0, 0))

        t2 = self.train_data.index / self.fs  # this only occurs when self.train_data is a dataframe
        t2 = t2[:len(t2)-125]  # trim the last 250 ms to align

        pose_train = event[:, 0]  # train data
        pose_predict = event[:, 1]  # train data
        k = accuracy_score(pose_predict, pose_train)
        title = "Results using Deep Learning" "\n Accuracy: " + str(k * 100) + "%"
        self.analysisTab.POSE.plot(pose_predict, pen=None, symbol='o', symbolPen=None, symbolBrush=brush,
                                   symbolSize=1)
        self.analysisTab.POSE.setTitle(title)

    def patientPredictions(self, event):
        self.analysisTab.POSE.clear()
        brush = pg.mkBrush(color=(255, 0, 0))

        pose_predict = event
        t = np.linspace(0, len(pose_predict) / 500, num=len(pose_predict))
        title = "Testing patient using new prompt"
        self.analysisTab.POSE.plot(t, pose_predict, pen=None, symbol='o', symbolPen=None, symbolBrush=brush,
                                   symbolSize=1)
        self.analysisTab.POSE.setTitle(title)

    def saveModel(self, event):
        self.scaler = event[0]  # Fitted scaler
        if self.model is None:
            self.model = event[1]
            fname, done1 = QtWidgets.QInputDialog.getText(
                self, 'Create Model', 'Enter name for new model')
            model_path = f'{fname}/model'
            self.model.save(f"Models/{model_path}")
            scaler_path = f'Models/{fname}/scaler.save'
            joblib.dump(self.scaler, scaler_path)
            message = f'{fname} successfully trained!'
            self.statusbar.showMessage(message)
            self.mainTab.model_status.setText(str(fname))
        else:
            self.model = event[1]
            fname, done1 = QtWidgets.QInputDialog.getText(
                self, 'Update Model', 'Enter name for updated model')
            self.model.save(f"Models/{fname}")
            message = f'{fname} successfully retrained!'
            self.statusbar.showMessage(message)

    def time_progress(self, event):
        self.mainTab.countdown_label.setText(f"BEGINNING IN: {event}")
        self.patientUI.countdown_label.setText(f"BEGINNING IN: {event}")

    def updateTimer(self, event):
        timer = event
        self.mainTab.timer_label.setText(str(timer))

    def updateImg(self):
        if self.count == self.train_reps - 1:
            self.count = -1
            self.mainTab.pose_img.clear()
            self.patientUI.pose_img.clear()
            self.mainTab.countdown_label.clear()
            self.patientUI.countdown_label.clear()
        else:
            self.count += 1
            pose_name = list(settings.pos.keys())
            pose = self.prompt[self.count]
            self.mainTab.countdown_label.setText(pose_name[pose])
            self.patientUI.countdown_label.setText(pose_name[pose])
            self.mainTab.pose_img.setPixmap((QtGui.QPixmap(settings.imgs[pose])))
            self.patientUI.pose_img.setPixmap((QtGui.QPixmap(settings.imgs[pose])))

    def updateProgress(self, event):
        progress = int(event * 100)  # convert to int out of 100

        self.mainTab.pb.setValue(progress)
        self.patientUI.pb.setValue(progress)

    def validatePose(self, event):
        true_label = emg_classification.Class(event[0])
        predicted_label = emg_classification.Class(event[1])

        true_label = true_label.name.lower()
        predicted_label = predicted_label.name.lower()

        if true_label == predicted_label:
            message = f'Correct: Detected {predicted_label} for {true_label}'
        else:
            message = f'Incorrect: Detected {predicted_label} for {true_label}'
        self.mainTab.detection_label.setText(message)

    def collectData(self):
        # Create QThread objects
        self.collectThread = QtCore.QThread()
        self.countdownThread = QtCore.QThread()

        # Create worker objects
        self.collectWorker = CollectWorker(self.prompt, self.train_data, self.train_reps, self.board, self.collect_time)
        self.countdownWorker = CountdownWorker(5)  # countdown of five

        self.timerWorker = TimerWorker(self.collect_time, self.train_reps)
        self.timerThread = QtCore.QThread()

        # Move workers to the threads
        self.collectWorker.moveToThread(self.collectThread)
        self.countdownWorker.moveToThread(self.countdownThread)
        self.timerWorker.moveToThread(self.timerThread)
        # Connect signals and slots
        self.collectThread.started.connect(self.collectWorker.run)
        self.collectWorker.finished.connect(self.collectThread.quit)
        self.collectWorker.finished.connect(self.collectWorker.deleteLater)
        self.collectThread.finished.connect(self.collectThread.deleteLater)
        self.collectWorker.data.connect(self.saveDataFile)
        self.collectWorker.updateImg.connect(self.updateImg)
        self.collectWorker.updateProgress.connect(self.updateProgress)

        self.countdownThread.started.connect(self.countdownWorker.run)
        self.countdownWorker.finished.connect(self.collectThread.start)
        self.countdownWorker.finished.connect(self.timerThread.start)
        self.countdownWorker.finished.connect(self.countdownThread.quit)
        self.countdownWorker.finished.connect(self.countdownWorker.deleteLater)
        self.countdownThread.finished.connect(self.countdownThread.deleteLater)
        self.countdownThread.finished.connect(
            lambda: self.mainTab.countdown_label.clear()
        )
        self.countdownThread.finished.connect(
            lambda: self.patientUI.countdown_label.clear()
        )
        self.countdownThread.finished.connect(
            lambda: self.mainTab.pb.setVisible(True)
        )
        self.countdownThread.finished.connect(
            lambda: self.patientUI.pb.setVisible(True)
        )
        self.countdownWorker.updateTime.connect(self.time_progress)

        self.timerThread.started.connect(self.timerWorker.run)
        self.timerWorker.finished.connect(self.timerThread.quit)
        self.timerWorker.finished.connect(self.timerWorker.deleteLater)
        self.timerThread.finished.connect(self.timerThread.deleteLater)
        self.timerThread.finished.connect(
            lambda: self.mainTab.timer_label.clear()
        )
        self.timerWorker.updateTimer.connect(self.updateTimer)
        # Start the thread

        if len(self.prompt) > 0:
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
            lambda: self.patientUI.pb.setHidden(True)
        )
        self.collectThread.finished.connect(
            lambda: self.mainTab.pb.setValue(0)
        )
        self.collectThread.finished.connect(
            lambda: self.patientUI.pb.setValue(0)
        )

    def trainData(self):
        # Create QThread objects
        self.trainThread = QtCore.QThread()

        # Create worker objects
        self.trainWorker = TrainWorker(self.td_selchan, self.model, self.scaler)  # specify the channels to use to train

        # Move workers to the threads
        self.trainWorker.moveToThread(self.trainThread)

        # Connect signals and slots
        self.trainThread.started.connect(self.trainWorker.run)
        self.trainWorker.finished.connect(self.trainThread.quit)
        self.trainWorker.finished.connect(self.trainWorker.deleteLater)
        self.trainThread.finished.connect(self.trainThread.deleteLater)
        self.trainWorker.paramSignal.connect(self.saveModel)
        self.trainWorker.dataSignal.connect(self.dataPrediction)

        # Start the thread

        if len(self.train_data) == 0:
            message = 'Cannot train model with data length of 0'
            QtWidgets.QMessageBox.warning(self, 'Help', message)
        else:
            self.trainThread.start()
            self.mainTab.collect.setEnabled(False)
            self.mainTab.train.setEnabled(False)
            self.mainTab.test.setEnabled(False)

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

    def testData(self):
        # Create QThread objects
        self.testThread = QtCore.QThread()
        self.countdownThread = QtCore.QThread()

        if self.serial_obj is not None:
            # Create worker objects
            # Must use the selected channels after training
            selected_channels = list(self.td_selchan)
            if 'Pose' in selected_channels:
                selected_channels.remove('Pose')
            self.testWorker = TestWorker(self.prompt, selected_channels, self.train_reps,
                                         self.model, self.scaler, self.board, self.serial_obj, self.collect_time)
            self.countdownWorker = CountdownWorker(5)
            # Move workers to the threads
            self.testWorker.moveToThread(self.testThread)
            self.countdownWorker.moveToThread(self.countdownThread)
            # Connect signals and slots
            self.testThread.started.connect(self.testWorker.run)
            self.testWorker.finished.connect(self.testThread.quit)
            self.testWorker.finished.connect(self.testWorker.deleteLater)
            self.testThread.finished.connect(self.testThread.deleteLater)
            self.testWorker.updateImg.connect(self.updateImg)
            self.testWorker.predictions.connect(self.patientPredictions)
            self.testWorker.validatePose.connect(self.validatePose)

            self.countdownThread.started.connect(self.countdownWorker.run)
            self.countdownWorker.finished.connect(self.testThread.start)
            self.countdownWorker.finished.connect(self.countdownThread.quit)
            self.countdownWorker.finished.connect(self.countdownWorker.deleteLater)
            self.countdownThread.finished.connect(self.countdownThread.deleteLater)
            self.countdownThread.finished.connect(
                lambda: self.mainTab.countdown_label.clear()
            )
            self.countdownThread.finished.connect(
                lambda: self.patientUI.countdown_label.clear()
            )

            self.countdownWorker.updateTime.connect(self.time_progress)

            # Start the thread
            if self.model is None:
                message = 'Cannot test a non-existent model'
                QtWidgets.QMessageBox.warning(self, 'Help', message)

            elif len(self.prompt) == 0:
                message = 'Insert a prompt, found under "Tools", to test model'
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
        else:
            message = 'Port is not connected'
            QtWidgets.QMessageBox.warning(self, 'Warning', message)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = PoseApp()
    sys.exit(app.exec())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
