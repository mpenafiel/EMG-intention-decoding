import datetime
import os
import sys
import time

import nidaqmx
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import auxilliary
import settings
from windows import createMain, createAnalysis, createPatientView


# Some icons by Yusuke Kamiyamane. Licensed under a Creative Commons Attribution 3.0 License.


def createBoard():
    params = MindRoveInputParams()
    board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD.value, params)
    return board


class DaqWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.is_killed = False

    def run(self):
        import nidaqmx.system
        system = nidaqmx.system.System.local()
        device = system.devices['Dev1']
        while True:
            try:
                device.self_test_device()
                self.status.emit(101)
            except nidaqmx.DaqError:
                self.status.emit(100)
            finally:
                if self.is_killed:
                    break
        self.finished.emit()

    def kill(self):
        self.is_killed = True


class CollectWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateImg = QtCore.pyqtSignal()
    data = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, prompt, train_data, train_reps, board):
        super().__init__()

        self.prompt = prompt
        self.train_data = train_data
        self.train_reps = train_reps
        self.board = board

    def stream(self):
        for i in range(self.train_reps):
            self.updateImg.emit()
            k = self.prompt[i]
            self.board.start_stream()
            time.sleep(4)
            self.board.stop_stream()
            data = self.board.get_board_data()

            # Processing Data
            data[8, :] = int(k)
            data = np.transpose(data)
            dataDF = pd.DataFrame(data[:, 0:9],
                                  columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])
            data = dataDF.reset_index(drop=True)
            original = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])
            filtered = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])
            smoothed = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])

            fs = 500
            fhc = 200
            flc = 4
            ws = 50

            for j in filtered.columns:
                original[j] = data[j]
                filtered[j] = settings.butter_bandpass_filter(data[j], flc, fhc, fs, order=6)
                smoothed[j] = settings.window_rms(filtered[j], ws)

            print(len(smoothed))
            print(len(data.Pose))

            smoothed.Pose = data.Pose
            self.train_data = np.concatenate((self.train_data, smoothed))
        self.updateImg.emit()
        self.train_data = pd.DataFrame(self.train_data,
                                       columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])
        self.board.release_session()
        self.finished.emit()
        self.data.emit(self.train_data)

    def run(self):
        if self.board.is_prepared():
            self.stream()
        else:
            self.board.prepare_session()
            self.stream()


class TrainWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    modelSignal = QtCore.pyqtSignal(Sequential)
    dataSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, train_data, model, pose):
        super().__init__()
        self.train_data = train_data
        self.model = model
        self.pose = pose

    def run(self):
        # From sklearn.metrics import classification_report
        X = self.train_data[["CH0", "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7"]]
        y = settings.get_one_hot(self.train_data.Pose.astype(int), 3)

        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

        if self.model is None:
            self.model = Sequential()
            self.model.add(Dense(2500, activation='relu', input_shape=(8,)))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(3, activation="softmax"))

            self.model.compile(loss='categorical_crossentropy',
                               optimizer='Adam',
                               metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=2, batch_size=2, verbose=1)

        y_predicted = self.model.predict(X)

        y_predicted[y_predicted[:, :] > 0.7] = 1
        y_predicted[y_predicted[:, :] != 1] = 0

        # create data frame to easily manipulate values
        y_predicted_df = pd.DataFrame(y_predicted)
        y_df = pd.DataFrame(y)

        pose_predict = np.zeros((len(y_predicted),), dtype=int)
        pose_train = np.zeros((len(y_predicted),), dtype=int)

        for (columnName, columnData) in y_predicted_df.items():
            column = columnData.values.reshape(columnData.values.shape[0], ) * columnName
            pose_predict = pose_predict + column

        for (columnName, columnData) in y_df.items():
            column = columnData.values.reshape(columnData.values.shape[0], ) * columnName
            pose_train = pose_train + column

        data = np.column_stack((pose_train, pose_predict))  # first column [0] -> train ... second column [1] -> predict

        self.modelSignal.emit(self.model)  # Keras model
        self.dataSignal.emit(data)  # numpy 2-D array of poses
        self.finished.emit()


class TestWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateImg = QtCore.pyqtSignal()
    DAQerror = QtCore.pyqtSignal()

    def __init__(self, prompt, train_data, train_reps, model, board):
        super().__init__()

        self.prompt = prompt
        self.train_data = train_data
        self.train_reps = train_reps
        self.model = model
        self.board = board

    def test(self):

        try:
            task_write = nidaqmx.task.Task("volOUT")
            task_write.ao_channels.add_ao_voltage_chan("Dev1/ao0", 'TIM', 0, 3.3)
            task_write.ao_channels.add_ao_voltage_chan("Dev1/ao1", 'RL', 0, 3.3)

            #  timenow = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

            task_write.start()

            poses = list()

            fs = 500
            fhc = 200
            flc = 4
            ws = 50

            for i in range(self.train_reps):
                self.updateImg.emit()
                k = self.prompt[i]
                self.board.start_stream()
                time.sleep(4)
                self.board.stop_stream()
                data = self.board.get_board_data()

                # Processing Data
                data[8, :] = int(k)
                data = np.transpose(data)
                dataDF = pd.DataFrame(data[:, 0:9],
                                      columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])
                data = dataDF.reset_index(drop=True)

                original = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])
                filtered = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7'])
                smoothed = pd.DataFrame(columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])

                for j in filtered.columns:
                    original[j] = data[j]
                    filtered[j] = settings.butter_bandpass_filter(data[j], flc, fhc, fs, order=6)
                    smoothed[j] = settings.window_rms(filtered[j], ws)

                smoothed.Pose = data.Pose

                X = smoothed[["CH0", "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7"]]
                X = StandardScaler().fit_transform(X)
                Y = self.model.predict(X)

                Y[Y[:, :] > 0.7] = 1
                Y[Y[:, :] != 1] = 0
                pose = np.argmax(np.count_nonzero(Y, axis=0))
                print(pose)

                vals = settings.pos_list[pose]
                task_write.write(vals)
                tim_val = round(vals[0], 5)
                rl_val = round(vals[1], 5)
                print(f"TIM: {tim_val}, RL: {rl_val}")
                time.sleep(2)

                vals = settings.pos_list[0]
                task_write.write(vals)

                poses.append(pose)
                self.train_data = np.concatenate((self.train_data, smoothed))
            print(poses)
            self.updateImg.emit()
            task_write.stop()
            task_write.close()

            self.train_data = pd.DataFrame(self.train_data,
                                           columns=['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'Pose'])
            # self.train_data.to_csv(f"EMG_data/emg_data{timenow}.csv")
            self.board.release_session()
            self.finished.emit()
        except nidaqmx.DaqError:
            self.DAQerror.emit()

    def run(self):
        if self.board.is_prepared():
            self.test()
        else:
            self.board.prepare_session()
            self.test()


class PoseApp(QtWidgets.QMainWindow):
    count = -1
    prompt = []
    train_reps = len(prompt)
    train_data = np.ndarray(shape=(0, 9), dtype=np.int64)
    test_data = None
    model = None
    table = None
    board = createBoard()

    def __init__(self):
        # Call the Parent constructor
        super().__init__()

        self.daqThread = QtCore.QThread()
        self.daqWorker = DaqWorker()

        self.setWindowTitle("EMG Pose Classifications")
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
        self._daqStatus()
        #  self.mindRoveStatus()

        self.mainTab.collect.pressed.connect(self.collectData)
        self.mainTab.train.pressed.connect(self.trainData)
        self.mainTab.test.pressed.connect(self.testData)

        self.channels = [self.analysisTab.CH0,
                         self.analysisTab.CH1,
                         self.analysisTab.CH2,
                         self.analysisTab.CH3,
                         self.analysisTab.CH4,
                         self.analysisTab.CH5,
                         self.analysisTab.CH6,
                         self.analysisTab.CH7]

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
        toolMenu.addAction(self.configureDaqAction)
        # Help menu
        helpMenu = menuBar.addMenu("Help")
        helpMenu.addAction(self.documentationAction)
        helpMenu.addAction(self.aboutAction)

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
        self.newPromptAction = QtGui.QAction("&New Prompt...", self)
        self.randPromptAction = QtGui.QAction("&Random Prompt", self)
        self.randLenPromptAction = QtGui.QAction("Random &Length Prompt", self)
        self.configureDaqAction = QtGui.QAction("Configure DAQ", self)

    def _connectActions(self):
        # Connect File actions
        self.loadDataAction.triggered.connect(self.loadData)
        self.loadModelAction.triggered.connect(self.loadModel)
        self.clearDataAction.triggered.connect(self.clearData)
        self.clearModelAction.triggered.connect(self.clearModel)
        self.clearAllAction.triggered.connect(self.clearAll)
        self.newPromptAction.triggered.connect(self.newPrompt)
        self.randPromptAction.triggered.connect(self.randPrompt)
        self.randLenPromptAction.triggered.connect(self.randLenPrompt)
        self.configureDaqAction.triggered.connect(self.configureDAQ)
        self.exitAction.triggered.connect(self.close)

        # Connect Help actions
        self.documentationAction.triggered.connect(auxilliary.documentation)
        self.aboutAction.triggered.connect(self.about)

    def _createStatusBar(self):
        self.statusbar = self.statusBar()
        # Adding a temporary message
        self.statusbar.showMessage("Ready", 3000)

    def loadData(self):
        fname, selFilter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                                 'c:\\', "CSV files(*.csv);;XML files(*.xml);;Text "
                                                                         "files (*.txt)")
        self.train_data = pd.read_csv(fname)
        message = f'{fname} successfully uploaded!'
        self.statusbar.showMessage(message)
        self.mainTab.data_status.setText('Loaded')
        self.updateData()

    def clearData(self):
        self.train_data = np.ndarray(shape=(0, 9), dtype=np.int64)
        self.mainTab.data_status.setText('')

    def loadModel(self):

        folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Model"))
        self.model = keras.models.load_model(folder)
        message = f'{folder} successfully uploaded!'
        self.statusbar.showMessage(message)
        self.mainTab.model_status.setText('Loaded')

    def clearModel(self):
        self.model = None
        self.mainTab.model_status.setText('')

    def clearAll(self):
        self.train_data = np.ndarray(shape=(0, 9), dtype=np.int64)
        self.model = None
        self.mainTab.data_status.setText('')
        self.mainTab.model_status.setText('')

    def newPrompt(self):
        # Generate a new prompt with any Pose assignment between 0 and 20

        while True:
            new_prompt, done1 = QtWidgets.QInputDialog.getText(
                self, 'Input Dialogue', 'Enter new Prompt:')
            new_prompt = new_prompt.split()
            for i in range(len(new_prompt)):
                if new_prompt[i].isdigit():
                    new_prompt[i] = int(new_prompt[i])

            if all([isinstance(item, int) for item in new_prompt]):
                if all([int(item) >= 0 for item in new_prompt]) and all([int(item) <= 4 for item in new_prompt]):
                    if len(new_prompt) <= 20:
                        self.prompt = new_prompt
                        self.mainTab.prompt_status.setText(f'{self.prompt}')
                        self.train_reps = len(self.prompt)
                        break
                    else:
                        QtWidgets.QMessageBox.information(self, 'Help', 'Prompt length should be no more than 20')
                        continue
                else:
                    QtWidgets.QMessageBox.information(self, 'Help', 'Integers should be between 0 and 4')
                    continue
            else:
                QtWidgets.QMessageBox.information(self, 'Help', 'Create input as integers separated by space')
                continue

    def randPrompt(self):
        # Generate a randomized prompt of any length between 0 and 20

        length, done1 = QtWidgets.QInputDialog.getInt(
            self, 'Assign a Random Prompt', 'Enter the number of poses you want to implement (maximum of 20 allowed):',
            min=0, max=20)

        new_prompt = np.random.randint(0, high=5, size=(1, int(length)))
        new_prompt = new_prompt.tolist()  # converts into a list within a list
        self.prompt = new_prompt[0]  # obtain the list
        self.train_reps = len(self.prompt)
        self.mainTab.prompt_status.setText(f'{self.prompt}')

    def randLenPrompt(self):
        # Generate a pseudorandom prompt up to 20 poses in length
        length = np.random.randint(4, high=21)
        new_prompt = np.random.randint(0, high=5, size=(1, length))
        new_prompt = new_prompt.tolist()
        self.prompt = new_prompt[0]
        self.train_reps = len(self.prompt)
        self.mainTab.prompt_status.setText(f'{self.prompt}')

    def configureDAQ(self):
        pass

    def about(self):
        # Logic for showing an 'about' dialog content goes here...
        QtWidgets.QMessageBox.information(self, 'Pose Classification', '')

    def closeEvent(self, event):
        if self.board.is_prepared():
            close = QtWidgets.QMessageBox.question(self,
                                                   "QUIT",
                                                   "ARE YOU SURE? Quitting will interrupt data collection",
                                                   QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if close == QtWidgets.QMessageBox.StandardButton.Yes:  # add boolean statement for threads
                self.daqWorker.kill()
                self.board.release_session()
                self.patientUI.close()
                event.accept()
            elif close == QtWidgets.QMessageBox.StandardButton.No:
                event.ignore()
        else:
            self.daqWorker.kill()
            self.patientUI.close()
            event.accept()

    def changeImg(self):
        if self.count == self.train_reps - 1:
            self.count = -1
            self.mainTab.image.clear()
            self.patientUI.image.clear()
        else:
            self.count += 1
            self.mainTab.image.setPixmap((QtGui.QPixmap(settings.imgs[self.prompt[self.count]])))
            self.patientUI.image.setPixmap((QtGui.QPixmap(settings.imgs[self.prompt[self.count]])))

    def reportStatus(self, event):
        if event == 101:
            self.mainTab.daq_status.setText('Ready')
        else:
            self.mainTab.daq_status.setText('Not Ready')

    def saveDataFile(self, event):
        time_now = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        self.train_data = event
        filename = QtWidgets.QMessageBox.question(QtWidgets.QWidget(), "Save Data", "Create new file name.",
                                                  QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        home_dir = str(os.path.expanduser('~'))
        fpath = str
        if filename == QtWidgets.QMessageBox.StandardButton.Yes:
            [fpath, _] = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', home_dir, "CSV files(*.csv)")
        elif filename == QtWidgets.QMessageBox.StandardButton.No:
            fpath = f"{home_dir}/emg_data{time_now}.csv"
        self.train_data.to_csv(fpath, index=False)
        message = f'Successfully collected data and saved to {fpath}!'
        self.statusbar.showMessage(message)
        self.mainTab.data_status.setText('Loaded')
        self.updateData()

    def updateData(self):
        t = np.linspace(1, len(self.train_data.CH1) / 500, num=len(self.train_data.CH1))
        for channel in self.channels:
            index = self.channels.index(channel)
            channel.plot(t, self.train_data.iloc[:, index])  # for future reference, change how data is written so
            # the first column doesn't have the index
        self.analysisTab.POSE.plot(t, self.train_data.iloc[:, -1])

        self.table = auxilliary.TableModel(self.train_data)
        self.analysisTab.table.setModel(self.table)

    def dataPrediction(self, event):
        brush = pg.mkBrush(color=(255, 0, 0))

        t = np.linspace(1, len(self.train_data.CH1) / 500, num=len(self.train_data.CH1))
        pose_train = event[:, 0]  # train data
        pose_predict = event[:, 1]  # train data
        k = accuracy_score(pose_predict, pose_train)
        title = "Results using Deep Learning" "\n Accuracy: " + str(k * 100) + "%"
        self.analysisTab.POSE.plot(t, pose_predict, pen=None, symbol='o', symbolPen=None, symbolBrush=brush,
                                   symbolSize=6)
        self.analysisTab.POSE.setTitle(title)

    def saveModel(self, event):
        if self.model is None:
            self.model = event
            fname, done1 = QtWidgets.QInputDialog.getText(
                self, 'Create Model', 'Enter name for new model')
            self.model.save(f"Models/{fname}")
            message = f'{fname} successfully trained!'
            self.statusbar.showMessage(message)
        else:
            self.model = event
            fname, done1 = QtWidgets.QInputDialog.getText(
                self, 'Update Model', 'Enter name for updated model')
            self.model.save(f"Models/{fname}")
            message = f'{fname} successfully retrained!'
            self.statusbar.showMessage(message)

    def _daqStatus(self):

        self.daqWorker.moveToThread(self.daqThread)
        self.daqThread.started.connect(self.daqWorker.run)
        self.daqWorker.finished.connect(self.daqThread.quit)
        self.daqWorker.finished.connect(self.daqWorker.deleteLater)
        self.daqThread.finished.connect(self.daqThread.deleteLater)
        self.daqWorker.status.connect(self.reportStatus)

        # Start the thread
        self.daqThread.start()

    def collectData(self):
        # Create QThread objects
        self.collectThread = QtCore.QThread()

        # Create worker objects
        self.collectWorker = CollectWorker(self.prompt, self.train_data, self.train_reps, self.board)

        # Move workers to the threads
        self.collectWorker.moveToThread(self.collectThread)
        # Connect signals and slots
        self.collectThread.started.connect(self.collectWorker.run)
        self.collectWorker.finished.connect(self.collectThread.quit)
        self.collectWorker.finished.connect(self.collectWorker.deleteLater)
        self.collectThread.finished.connect(self.collectThread.deleteLater)
        self.collectWorker.data.connect(self.saveDataFile)
        self.collectWorker.updateImg.connect(self.changeImg)

        # Start the thread

        if len(self.prompt) > 0:
            self.collectThread.start()
            self.mainTab.collect.setEnabled(False)
            self.mainTab.train.setEnabled(False)
            self.mainTab.test.setEnabled(False)
        else:
            QtWidgets.QMessageBox.information(self, 'Help', 'Insert a prompt, found under "Tools", to begin '
                                                            'collecting data')

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

    def trainData(self):
        # Create QThread objects
        self.trainThread = QtCore.QThread()

        # Create worker objects
        self.trainWorker = TrainWorker(self.train_data, self.model, self.analysisTab.POSE)

        # Move workers to the threads
        self.trainWorker.moveToThread(self.trainThread)

        # Connect signals and slots
        self.trainThread.started.connect(self.trainWorker.run)
        self.trainWorker.finished.connect(self.trainThread.quit)
        self.trainWorker.finished.connect(self.trainWorker.deleteLater)
        self.trainThread.finished.connect(self.trainThread.deleteLater)
        self.trainWorker.modelSignal.connect(self.saveModel)
        self.trainWorker.dataSignal.connect(self.dataPrediction)

        # Start the thread

        if len(self.train_data) == 0:
            QtWidgets.QMessageBox.warning(self, 'Help', 'CANNOT TRAIN MODEL WITH DATA ARRAY OF LENGTH 0')
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

        # Create worker objects
        self.testWorker = TestWorker(self.prompt, self.train_data, self.train_reps, self.model, self.board)

        # Move workers to the threads
        self.testWorker.moveToThread(self.testThread)
        # Connect signals and slots
        self.testThread.started.connect(self.testWorker.run)
        self.testWorker.finished.connect(self.testThread.quit)
        self.testWorker.finished.connect(self.testWorker.deleteLater)
        self.testThread.finished.connect(self.testThread.deleteLater)
        self.testWorker.updateImg.connect(self.changeImg)

        self.testWorker.DAQerror.connect(
            lambda: QtWidgets.QMessageBox.warning(self, 'Error', """DaqError: The specified device is not present or 
            is not active in the system. The device may not be installed on this system, may have been unplugged, 
            or may not be installed correctly.""")
        )
        # Reset tasks
        self.testWorker.DAQerror.connect(self.testWorker.deleteLater)
        self.testWorker.DAQerror.connect(
            lambda: self.mainTab.collect.setEnabled(True)
        )
        self.testWorker.DAQerror.connect(
            lambda: self.mainTab.train.setEnabled(True)
        )
        self.testWorker.DAQerror.connect(
            lambda: self.mainTab.test.setEnabled(True)
        )

        # Start the thread
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, 'Help', 'Cannot test a non-existent model')

        elif len(self.prompt) == 0:
            QtWidgets.QMessageBox.information(self, 'Help', 'Insert a prompt, found under "Tools", to test model')
        else:
            self.testThread.start()
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = PoseApp()
    sys.exit(app.exec())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
