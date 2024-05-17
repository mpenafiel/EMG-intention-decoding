from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import os
import matplotlib as mpl
mpl.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import utils
import numpy as np
import config
import time
import config

class DemoWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    updateCountdown = QtCore.pyqtSignal(int)
    updateTimer = QtCore.pyqtSignal(int)
    updateProgress = QtCore.pyqtSignal(float)
    updateImg = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.active_timer = 3 # seconds
        self.inactive_timer = 6 # seconds
        self.tasks = [0, 1, 0, 3, 0, 4, 0, 2, 0, 5, 0, 6]
        self.countdown = 5 # seconds
        self.run_flag = True
    
    def interruptDemo(self):
        self.run_flag = False

    def run(self):
        while self.run_flag:
            while self.countdown and self.run_flag:
                self.updateCountdown.emit(self.countdown)
                time.sleep(1)
                self.countdown -= 1
            for i in range(len(self.tasks)):
                k = int(self.tasks[i])
                self.updateImg.emit(k) # Emit task value
                if k == 0: # Inctive interval
                    timer = self.inactive_timer
                elif k != 0: # Active interval
                    timer = self.active_timer
                while timer and self.run_flag:
                    self.updateTimer.emit(timer)
                    time.sleep(1)
                    timer -= 1
                progress = (i + 1) / len(self.tasks)
                self.updateProgress.emit(progress)
            self.run_flag = False
        self.updateImg.emit(-1)
        self.finished.emit()

class SessionData(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Save File')
        
        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.subject_label = QtWidgets.QLabel('Subject: ', self)
        self.subject_label.setStyleSheet("font-weight: bold; color: black")
        if parent.subject_path is None:
            self.subject_name = QtWidgets.QLineEdit(self)
            self.subject_name.setPlaceholderText('Enter Subject Name')
        else:
            self.subject_path = parent.subject_path
            self.subject_name = QtWidgets.QLineEdit(os.path.basename(parent.subject_path), self)

        self.subject_name.setReadOnly(True)

        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/folder-open.png'))
        self.open_subject_folder = QtWidgets.QToolButton()
        self.open_subject_folder.setIcon(icon)
        self.open_subject_folder.setAutoRaise(False)
        newTip = "Open an existing subject's folder or create new one"
        self.open_subject_folder.setToolTip(newTip)

        self.set_fname = QtWidgets.QCheckBox('Create new file name. ', self)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.subject_label, 0, 0)
        layout.addWidget(self.subject_name, 0, 1, 1, 2)
        layout.addWidget(self.open_subject_folder, 0, 3)
        layout.addWidget(self.set_fname, 1, 0, 1, 4)
        layout.addWidget(self.buttonBox, 2, 0, 1, 4)

        self.setLayout(layout)
        self.show()
        self.setFixedSize(self.size())

        self.open_subject_folder.clicked.connect(self.openSubjectFolder)
    
    def openSubjectFolder(self):
        self.subject_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Subject",
                                                                  options=QtWidgets.QFileDialog.Option.ReadOnly | QtWidgets.QFileDialog.Option.ShowDirsOnly)
        if self.subject_path:
            subject_name = os.path.basename(self.subject_path)
            self.subject_name.setText(subject_name)


class SessionModel(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Create Model')
        
        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.subject_label = QtWidgets.QLabel('Subject: ', self)
        self.subject_label.setStyleSheet("font-weight: bold; color: black")
        if parent.subject_path is None:
            self.subject_name = QtWidgets.QLineEdit(self)
            self.subject_name.setPlaceholderText('Enter Subject Name')
        else:
            self.subject_path = parent.subject_path
            self.subject_name = QtWidgets.QLineEdit(os.path.basename(parent.subject_path), self)

        self.subject_name.setReadOnly(True)

        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/folder-open.png'))
        self.open_subject_folder = QtWidgets.QToolButton()
        self.open_subject_folder.setIcon(icon)
        self.open_subject_folder.setAutoRaise(False)
        newTip = "Open an existing subject's folder or create a new one"
        self.open_subject_folder.setToolTip(newTip)
        
        self.model_label = QtWidgets.QLabel('Model: ', self)
        self.model_label.setStyleSheet("font-weight: bold; color: black")
        self.model_name = QtWidgets.QLineEdit(self)
        self.model_name.setPlaceholderText('Enter Model Name')

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.subject_label, 0, 0)
        layout.addWidget(self.subject_name, 0, 1, 1, 2)
        layout.addWidget(self.open_subject_folder, 0, 3)
        layout.addWidget(self.model_label, 1, 0)
        layout.addWidget(self.model_name, 1, 1, 1, 3)
        layout.addWidget(self.buttonBox, 2, 0, 1, 4)

        self.setLayout(layout)
        self.show()
        self.setFixedSize(self.size())

        self.open_subject_folder.clicked.connect(self.createOpenSubjectFolder)
    
    def createOpenSubjectFolder(self):
        self.subject_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Subject",
                                                                  options=QtWidgets.QFileDialog.Option.ShowDirsOnly)
        if self.subject_path:
            subject_name = os.path.basename(self.subject_path)
            self.subject_name.setText(subject_name)

class ParameterDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Training Parameters')

        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.open_task = QtWidgets.QCheckBox('Open', self)
        self.close_task = QtWidgets.QCheckBox('Close', self)
        self.tripod_open_task = QtWidgets.QCheckBox('Tripod Open', self)
        self.tripod_pinch_task = QtWidgets.QCheckBox('Tripod Pinch', self)
        self.bottom_close_task = QtWidgets.QCheckBox('Bottom Close', self)
        self.bottom_open_task = QtWidgets.QCheckBox('Bottom Open', self)

        self.repetitions = QtWidgets.QSpinBox()
        self.repetitions.setMaximum(30)
        self.repetitions.setMinimum(3)

        self.rep_label = QtWidgets.QLabel('Repetitions for each task', self)

        self.task_frame = QtWidgets.QGroupBox('Training Tasks')

        layout1 = QtWidgets.QVBoxLayout()
        layout1.addWidget(self.open_task)
        layout1.addWidget(self.close_task)
        layout1.addWidget(self.tripod_open_task)
        layout1.addWidget(self.tripod_pinch_task)
        layout1.addWidget(self.bottom_open_task)
        layout1.addWidget(self.bottom_close_task)        

        self.task_frame.setLayout(layout1)

        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.rep_label)
        layout2.addWidget(self.repetitions)

        layout3 = QtWidgets.QVBoxLayout()
        layout3.addWidget(self.task_frame)
        layout3.addLayout(layout2)
        layout3.addWidget(self.buttonBox)

        self.setLayout(layout3)
        self.show()
        self.setFixedSize(self.size())

        #  create a dictionary with the associated values
        self.tasks = {self.open_task: 1,
                      self.close_task: 2,
                      self.tripod_open_task: 3,
                      self.tripod_pinch_task: 4,
                      self.bottom_open_task: 5,
                      self.bottom_close_task: 6}
        
class NewTaskDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('New Task Sequence')
        self.length = 0

        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.open_task_label = QtWidgets.QLabel('Open', self)
        self.close_task_label = QtWidgets.QLabel('Close', self)
        self.tripod_open_task_label = QtWidgets.QLabel('Tripod Open', self)
        self.tripod_pinch_task_label = QtWidgets.QLabel('Tripod Pinch', self)
        self.bottom_open_task_label = QtWidgets.QLabel('Bottom Open', self)
        self.bottom_close_task_label = QtWidgets.QLabel('Bottom Close', self)

        self.open_task = QtWidgets.QSpinBox()
        self.open_task.setRange(0,30)
        self.open_task.valueChanged.connect(self.computeTotalTasks)

        self.close_task = QtWidgets.QSpinBox()
        self.close_task.setRange(0, 30)
        self.close_task.valueChanged.connect(self.computeTotalTasks)

        self.tripod_open_task = QtWidgets.QSpinBox()
        self.tripod_open_task.setRange(0, 30)
        self.tripod_open_task.valueChanged.connect(self.computeTotalTasks)

        self.tripod_pinch_task = QtWidgets.QSpinBox()
        self.tripod_pinch_task.setRange(0, 30)
        self.tripod_pinch_task.valueChanged.connect(self.computeTotalTasks)

        self.bottom_open_task = QtWidgets.QSpinBox()
        self.bottom_open_task.setRange(0, 30)
        self.close_task.valueChanged.connect(self.computeTotalTasks)

        self.bottom_close_task = QtWidgets.QSpinBox()
        self.bottom_close_task.setRange(0, 30)
        self.bottom_close_task.valueChanged.connect(self.computeTotalTasks)

        self.poses_frame = QtWidgets.QGroupBox('Training Tasks')

        layout1 = QtWidgets.QGridLayout()
        layout1.addWidget(self.open_task_label, 0, 0)
        layout1.addWidget(self.close_task_label, 1, 0)
        layout1.addWidget(self.tripod_open_task_label, 0, 2)
        layout1.addWidget(self.tripod_pinch_task_label, 1, 2)
        layout1.addWidget(self.bottom_open_task_label, 0, 4)
        layout1.addWidget(self.bottom_close_task_label, 1, 4)

        layout1.addWidget(self.open_task, 0, 1)
        layout1.addWidget(self.close_task, 1, 1) 
        layout1.addWidget(self.tripod_open_task, 0, 3) 
        layout1.addWidget(self.tripod_pinch_task, 1, 3) 
        layout1.addWidget(self.bottom_open_task, 0, 5) 
        layout1.addWidget(self.bottom_close_task, 1, 5) 

        self.poses_frame.setLayout(layout1)

        layout2 = QtWidgets.QHBoxLayout()

        self.shuffle = QtWidgets.QCheckBox('Shuffle Tasks', self)
        self.length_label = QtWidgets.QLabel(f'Number of Tasks: {self.length}')

        layout2.addWidget(self.shuffle)
        layout2.addWidget(self.length_label)

        layout3 = QtWidgets.QVBoxLayout()
        layout3.addWidget(self.poses_frame)
        layout3.addLayout(layout2)
        layout3.addWidget(self.buttonBox)

        self.setLayout(layout3)
        self.show()
        self.setFixedSize(self.size())

        #  create a dictionary with the associated values
        self.tasks = {self.open_task: 1,
                      self.close_task: 2,
                      self.tripod_open_task: 3,
                      self.tripod_pinch_task: 4,
                      self.bottom_open_task: 5,
                      self.bottom_close_task: 6}
        
    def computeTotalTasks(self):
        self.length = 0
        for key in self.tasks.keys():
            self.length = self.length + key.value()
        self.length_label.setText(f'Number of Tasks: {self.length}')

class RandTaskDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Random Task Sequence')
        self.length = 0

        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.set_length = QtWidgets.QCheckBox('Assign Length: ', self)
        self.set_length.stateChanged.connect(self.checkState)

        self.length = QtWidgets.QSpinBox()
        self.length.setRange(0,30)
        self.length.setHidden(True)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.set_length, 0, 0)
        layout.addWidget(self.length, 0, 1)
        layout.addWidget(self.buttonBox, 1, 0, 1, 2)

        self.setLayout(layout)
        self.show()
        self.setFixedSize(self.size())
    
    def checkState(self):
        if self.set_length.isChecked():
            self.length.show()
        else:
            self.length.setHidden(True)

class TimeIntervalDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Assign New Time Interval')

        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        current_label = QtWidgets.QLabel(f'Current interval')
        current_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignCenter)
        current_label.setStyleSheet("font-weight: bold; color: black")

        if parent.active_time == parent.inactive_time:
            self.current_interval = QtWidgets.QLabel(f'Active/Inactive: {parent.active_time}')
        else:
            self.current_interval = QtWidgets.QLabel(f'Active: {parent.active_time}, Inactive: {parent.inactive_time}')

        self.assign_inactive = QtWidgets.QCheckBox('Assign Inactive Interval? ', self)
        self.assign_inactive.stateChanged.connect(self.checkState)

        self.active_label = QtWidgets.QLabel('Active Interval: ')

        self.active = QtWidgets.QSpinBox()
        self.active.setRange(2,10)

        self.inactive_label = QtWidgets.QLabel('Inactive Interval: ')
        self.inactive_label.setEnabled(False)

        self.inactive = QtWidgets.QSpinBox()
        self.inactive.setRange(2,10)
        self.inactive.setEnabled(False)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(current_label, 0, 0)
        layout.addWidget(self.current_interval, 0, 1)
        layout.addWidget(self.assign_inactive, 1, 0, 1, 2)
        layout.addWidget(self.active_label, 2, 0)
        layout.addWidget(self.active, 2, 1)
        layout.addWidget(self.inactive_label, 3, 0)
        layout.addWidget(self.inactive, 3, 1)

        self.layout2 = QtWidgets.QHBoxLayout()
        self.layout2.addWidget(self.inactive_label)
        self.layout2.addWidget(self.inactive)

        self.layout3 = QtWidgets.QVBoxLayout()
        self.layout3.addLayout(layout)
        self.layout3.addWidget(self.buttonBox)

        self.setLayout(self.layout3)

    def checkState(self):
        if self.assign_inactive.isChecked():
            # self.layout3.insertLayout(1, self.layout2)
            self.inactive_label.setEnabled(True)
            self.inactive.setEnabled(True)
        else:
            self.inactive_label.setEnabled(False)
            self.inactive.setEnabled(False)

class TrainingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Training Model')
        
        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.rejected.connect(parent.trainWorker.callbacks.end_training)

        self.acc_label = QtWidgets.QLabel('Accuary: ', self)
        self.acc_label.setStyleSheet("font-weight: bold; color: black")
        self.acc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.val_acc_label = QtWidgets.QLabel('Val_Accuracy: ', self)
        self.val_acc_label.setStyleSheet("font-weight: bold; color: black")
        self.val_acc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.loss_label = QtWidgets.QLabel('Loss: ', self)
        self.loss_label.setStyleSheet("font-weight: bold; color: black")
        self.loss_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.val_loss_label = QtWidgets.QLabel('Val_Loss: ', self)
        self.val_loss_label.setStyleSheet("font-weight: bold; color: black")
        self.val_loss_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.batch_label = QtWidgets.QLabel('Batch: ', self)
        self.batch_label.setStyleSheet("font-weight: bold; color: black")
        self.batch_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.epoch_label = QtWidgets.QLabel('Epoch: ', self)
        self.epoch_label.setStyleSheet("font-weight: bold; color: black")
        self.epoch_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.acc = QtWidgets.QLabel('0', self)
        self.val_acc = QtWidgets.QLabel('0', self)
        self.loss = QtWidgets.QLabel('0', self)
        self.val_loss = QtWidgets.QLabel('0', self)
        self.batch = QtWidgets.QLabel(self)
        self.epoch = QtWidgets.QLabel(self)

        self.message = QtWidgets.QLabel(self)
        self.message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignLeft)

        self.elapsed_time = QtWidgets.QLabel(self)
        self.elapsed_time.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom)
        
        self.training_pb = QtWidgets.QProgressBar()
        self.training_pb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.acc_label, 0, 0)
        layout.addWidget(self.acc, 0, 1)
        layout.addWidget(self.val_acc_label, 1, 0)
        layout.addWidget(self.val_acc, 1, 1)
        layout.addWidget(self.loss_label, 0, 2)
        layout.addWidget(self.loss, 0, 3)
        layout.addWidget(self.val_loss_label, 1, 2)
        layout.addWidget(self.val_loss, 1, 3)
        layout.addWidget(self.batch_label, 0, 4)
        layout.addWidget(self.batch, 0, 5)
        layout.addWidget(self.epoch_label, 1, 4)
        layout.addWidget(self.epoch, 1, 5)
        layout.addWidget(self.message, 2, 0, 1, 6)
        layout.addWidget(self.training_pb, 3, 0, 1, 6)
        layout.addWidget(self.elapsed_time, 4, 0, 1, 4)
        layout.addWidget(self.buttonBox, 4, 4, 1, 2)
        

        self.setLayout(layout)
        self.show()
        self.resize(360, 120)
        self.setFixedSize(self.size())

class ExampleWin(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Getting Started")
        self.setFixedSize(900, 600)

        id = QtGui.QFontDatabase.addApplicationFont(config.resource_path("dev/assets/fonts/TitilliumWeb-Bold.ttf"))
        families = QtGui.QFontDatabase.applicationFontFamilies(id)

        self.titleCard_title = QtWidgets.QLabel("Welcome to the Intention Detection System!", self)
        self.titleCard_title.setFont(QtGui.QFont(families[0], 24))
        self.titleCard_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.titleCard_title.setProperty('type', 5)
        
        file = open(config.resource_path('dev/assets/text/gettingStarted.txt'))
        text = file.read()
        file.close()

        self.help_description = QtWidgets.QLabel(text, self)
        self.help_description.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.help_description.setProperty('type', 6)
        self.help_description.setWordWrap(True)

        self.nextBtn = QtWidgets.QPushButton("Next", self)
        self.backBtn = QtWidgets.QPushButton("Back", self)

        self.nextBtn.setProperty("type", 1)
        self.backBtn.setProperty("type", 1)
        self.backBtn.setHidden(True)

        self.nextBtn.clicked.connect(self.nextCard)
        self.backBtn.clicked.connect(self.backCard)

        self.titleCard_layout = QtWidgets.QGridLayout()
        self.titleCard_layout.addWidget(self.titleCard_title, 0, 0, 1, 4)
        self.titleCard_layout.addWidget(self.help_description, 1, 0, 3, 4)

        self.titlePage = QtWidgets.QWidget()
        self.titlePage.setLayout(self.titleCard_layout)

        self.taskCard_title = QtWidgets.QLabel("Let's preview the Tasks", self)
        self.taskCard_title.setFont(QtGui.QFont(families[0], 24))
        self.taskCard_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.taskCard_title.setProperty('type', 5)

        self.task_label = QtWidgets.QLabel(self)
        self.task_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        tasksPixmap = QtGui.QPixmap(config.resource_path("dev/assets/imgs/tasks.png"))
        self.task_label.setPixmap(tasksPixmap)  

        self.taskCard_layout = QtWidgets.QGridLayout()
        self.taskCard_layout.addWidget(self.taskCard_title, 0, 0, 1, 4)
        self.taskCard_layout.addWidget(self.task_label, 1, 0, 5, 4)

        self.taskPage = QtWidgets.QWidget()
        self.taskPage.setLayout(self.taskCard_layout)

        self.demoCard_title = QtWidgets.QLabel("Demoing the Collect function", self)
        self.demoCard_title.setFont(QtGui.QFont(families[0], 24))
        self.demoCard_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.demoCard_title.setProperty('type', 5)

        file = open(config.resource_path('dev/assets/text/demoMsg.txt'))
        text = file.read()
        file.close()
        self.demo_description = QtWidgets.QLabel(text, self)
        self.demo_description.setProperty('type', 6)
        self.demo_description.setWordWrap(True)

        self.demo_label = QtWidgets.QLabel(self)
        self.demo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        demoPixmap = QtGui.QPixmap(config.resource_path("dev/assets/imgs/demo.png"))
        self.demo_label.setPixmap(demoPixmap)

        self.demoCard_layout = QtWidgets.QGridLayout()
        self.demoCard_layout.addWidget(self.demoCard_title, 0, 0, 1, 4)
        self.demoCard_layout.addWidget(self.demo_description, 1, 0, 2, 4)
        self.demoCard_layout.addWidget(self.demo_label, 3, 0, 4, 4)

        self.demoPage = QtWidgets.QWidget()
        self.demoPage.setLayout(self.demoCard_layout)

        self.practiceCard_title = QtWidgets.QLabel("Let's Practice", self)
        self.practiceCard_title.setFont(QtGui.QFont(families[0], 24))
        self.practiceCard_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.practiceCard_title.setProperty('type', 5)

        self.practice_countdown = QtWidgets.QLabel(self)
        self.practice_countdown.setFont(QtGui.QFont(families[0], 18))
        self.practice_countdown.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)
        self.practice_countdown.setProperty('type', 5)

        self.practice_taskLabel = QtWidgets.QLabel(self)
        self.practice_taskLabel.setFont(QtGui.QFont(families[0], 18))
        self.practice_taskLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.practice_taskLabel.setProperty('type', 5)

        self.practice_img = QtWidgets.QLabel(self)
        self.practice_img.setMinimumSize(1, 1)
        self.practice_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        
        self.practice_pixmap = QtGui.QPixmap()

        self.practice_collect = QtWidgets.QPushButton("Collect Data", self)
        self.practice_collect.setProperty('type', 1)

        self.practice_stop = QtWidgets.QPushButton("STOP", self)
        self.practice_stop.setProperty('type', 1)
        self.practice_stop.setHidden(True)

        self.practice_pb = QtWidgets.QProgressBar()
        self.practice_pb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.practice_pb.setHidden(True) 

        self.practiceCard_layout = QtWidgets.QGridLayout()
        self.practiceCard_layout.addWidget(self.practiceCard_title, 2, 0, 1, 3)
        self.practiceCard_layout.addWidget(self.practice_img, 0, 0, 6, 3)
        self.practiceCard_layout.addWidget(self.practice_countdown, 0, 2)
        self.practiceCard_layout.addWidget(self.practice_taskLabel, 0, 0)
        self.practiceCard_layout.addWidget(self.practice_pb, 7, 0, 1, 3)
        self.practiceCard_layout.addWidget(self.practice_collect, 8, 0)
        self.practiceCard_layout.addWidget(self.practice_stop, 8, 2)

        self.practicePage = QtWidgets.QWidget()
        self.practicePage.setLayout(self.practiceCard_layout)

        center = QtGui.QScreen.availableGeometry(QtWidgets.QApplication.primaryScreen()).center()
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.titlePage)
        self.stack.addWidget(self.taskPage)
        self.stack.addWidget(self.demoPage)
        self.stack.addWidget(self.practicePage)
        self.stack.setCurrentIndex(0)

        self.current_page_layout = QtWidgets.QGridLayout()
        self.current_page_layout.addWidget(self.stack, 0, 0, 4, 4)
        self.current_page_layout.addWidget(self.nextBtn, 4, 3)
        self.current_page_layout.addWidget(self.backBtn, 4, 0)

        self.current_page = QtWidgets.QWidget()
        self.current_page.setLayout(self.current_page_layout)

        self.setCentralWidget(self.current_page)

        self.page_id = 0
        self.practice_collect.clicked.connect(self.collectDemo)
        self.practice_stop.clicked.connect(self.interrupt)
        
        self.show()
        
    def nextCard(self):
        if self.page_id < self.stack.count()-1:
            self.page_id += 1
            if self.page_id == self.stack.count() - 1:
                self.nextBtn.setText("Finish")
            self.backBtn.setVisible(True)
            self.stack.setCurrentIndex(self.page_id)
        else:
            self.close()

    def backCard(self):
        self.page_id -= 1
        if self.page_id < self.stack.count() - 1:
                self.nextBtn.setText("Next")
        if self.page_id == 0:
            self.backBtn.setHidden(True)
        self.stack.setCurrentIndex(self.page_id)

    def interrupt(self):
        self.worker.interruptDemo()

    def updateCountdown(self, event):
        self.practice_taskLabel.setText(f"Beginning in: {event}")

    def updateImg(self, event):
        if event == -1:
            self.practice_img.clear()
            self.practice_taskLabel.clear()
            self.practice_countdown.clear()
            self.practice_pixmap = QtGui.QPixmap()
        else:
            task = event
            task_label = list(config.pos.keys())
            text = f'Task: {task_label[task]}'
            self.practice_taskLabel.setText(text)
            self.practice_pixmap = QtGui.QPixmap(config.imgs[task])
            w = self.practice_img.width()
            h = self.practice_img.height()
            self.practice_pixmap = self.practice_pixmap.scaledToHeight(h, QtCore.Qt.TransformationMode.FastTransformation)
            self.practice_img.setPixmap(self.practice_pixmap)

    def updateProgress(self, event):
        progress = int(event * 100)  # convert to int out of 100
        self.practice_pb.setValue(progress)

    def updateTimer(self, event):
        timer = event
        text = f'Time: {timer}'
        self.practice_countdown.setText(text)

    def collectDemo(self):
        # Create QThread objects
        self.thread = QtCore.QThread()

        # Create worker objects
        self.worker = DemoWorker()

        # Move workers to the threads
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.updateImg.connect(self.updateImg)
        self.worker.updateProgress.connect(self.updateProgress)
        self.worker.updateTimer.connect(self.updateTimer)
        self.worker.updateCountdown.connect(self.updateCountdown)

        # Start the thread
        self.practice_collect.setEnabled(False)
        self.practice_stop.setVisible(True)
        self.practice_pb.setVisible(True)
        self.practiceCard_title.setHidden(True)
        self.thread.start()
        

        # Reset tasks
        self.thread.finished.connect(
            lambda: self.practice_collect.setEnabled(True)
        )
        self.thread.finished.connect(
            lambda: self.practice_pb.setHidden(True)
        )
        self.thread.finished.connect(
            lambda: self.practice_pb.setValue(0)
        )
        self.thread.finished.connect(
            lambda: self.practiceCard_title.setVisible(True)
        )
        self.thread.finished.connect(
            lambda: self.practice_stop.setHidden(True)
        )


class ViewWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Window")
        # Set the geometry of the window
        self.setGeometry(500, 100, 1200, 800)

        self.task_img = QtWidgets.QLabel(self)
        self.task_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.countdown_label = QtWidgets.QLabel(self)
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        countdown_font = QtGui.QFont('Ariel', 30)
        countdown_font.setBold(True)
        self.countdown_label.setFont(countdown_font)
        self.pb = QtWidgets.QProgressBar()
        self.pb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pb.setHidden(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.countdown_label)
        layout.addWidget(self.task_img)
        layout.addWidget(self.pb)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

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


class AnalysisUI(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(AnalysisUI, self).__init__(parent)

        layout_win = QtWidgets.QGridLayout()

        self.layout_stack = QtWidgets.QStackedLayout()  # referenced in method

        self.CH1 = pg.PlotWidget(name='CH1')
        self.CH1.setLabel('left', 'Voltage', units='mV')
        self.CH1.setLabel('bottom', 'Time', units='s')
        self.CH1.setTitle("CH1")

        self.CH2 = pg.PlotWidget(name="CH2")
        self.CH2.setLabel('left', 'Voltage', units='mV')
        self.CH2.setLabel('bottom', 'Time', units='s')
        self.CH2.setTitle("CH2")

        self.CH3 = pg.PlotWidget(name='CH3')
        self.CH3.setLabel('left', 'Voltage', units='mV')
        self.CH3.setLabel('bottom', 'Time', units='s')
        self.CH3.setTitle("CH3")

        self.CH4 = pg.PlotWidget(name='CH4')
        self.CH4.setLabel('left', 'Voltage', units='mV')
        self.CH4.setLabel('bottom', 'Time', units='s')
        self.CH4.setTitle("CH4")

        self.CH5 = pg.PlotWidget(name="CH5")
        self.CH5.setLabel('left', 'Voltage', units='mV')
        self.CH5.setLabel('bottom', 'Time', units='s')
        self.CH5.setTitle("CH5")

        self.CH6 = pg.PlotWidget(name='CH6')
        self.CH6.setLabel('left', 'Voltage', units='mV')
        self.CH6.setLabel('bottom', 'Time', units='s')
        self.CH6.setTitle("CH6")

        self.CH7 = pg.PlotWidget(name='CH7')
        self.CH7.setLabel('left', 'Voltage', units='mV')
        self.CH7.setLabel('bottom', 'Time', units='s')
        self.CH7.setTitle("CH7")

        self.CH8 = pg.PlotWidget(name='CH8')
        self.CH8.setLabel('left', 'Voltage', units='mV')
        self.CH8.setLabel('bottom', 'Time', units='s')
        self.CH8.setTitle("CH8")

        layout_channels = QtWidgets.QGridLayout()

        layout_channels.addWidget(self.CH1, 0, 0)
        layout_channels.addWidget(self.CH2, 0, 1)
        layout_channels.addWidget(self.CH3, 1, 0)
        layout_channels.addWidget(self.CH4, 1, 1)
        layout_channels.addWidget(self.CH5, 2, 0)
        layout_channels.addWidget(self.CH6, 2, 1)
        layout_channels.addWidget(self.CH7, 3, 0)
        layout_channels.addWidget(self.CH8, 3, 1)

        channels_frame = QtWidgets.QWidget()
        channels_frame.setLayout(layout_channels)

        self.TASK = pg.PlotWidget(name='Tasks')
        self.TASK.setLabel('bottom', 'Time', units='s')
        position_labels = [
            # Generate a list of tuples (x_value, x_label)
            (0, 'Rest'), (1, 'Open'), (2, 'Close'), (3, 'Tripod Open'), (4, 'Tripod Pinch'), (5, 'Bottom Open'), (6, 'Bottom Close')
        ]
        ax = self.TASK.getAxis('left')
        # Pass the list in, *in* a list.
        ax.setTicks([position_labels])
        self.TASK.setTitle("Task")

        # Options to display various tabs
        self.table_button = QtWidgets.QRadioButton('Table', self)
        self.table_button.setProperty('type', 2)
        self.table_button.setCheckable(True)
        self.options_button = QtWidgets.QRadioButton('Configurations', self)
        self.options_button.setProperty('type', 2)
        self.options_button.setCheckable(True)
        self.stat_button = QtWidgets.QRadioButton('Visualization', self)
        self.stat_button.setProperty('type', 2)
        self.stat_button.setCheckable(True)

        self.options_group = QtWidgets.QButtonGroup() # Abstract button group
        self.options_group.addButton(self.options_button, id=0)
        self.options_button.setChecked(True)
        self.options_group.addButton(self.table_button, id=1)
        self.options_group.addButton(self.stat_button, id=2)
        self.options_group.buttonClicked.connect(self.activate_tab)

        layout1 = QtWidgets.QHBoxLayout()

        layout1.addWidget(self.options_button)
        layout1.addWidget(self.table_button)
        layout1.addWidget(self.stat_button)

        self.options_frame = QtWidgets.QGroupBox('Options')
        self.options_frame.setLayout(layout1)

        # Table Tab
        self.table = QtWidgets.QTableView()

        # Config Tab
        self.ch1_check = QtWidgets.QCheckBox('CH1', self)
        self.ch1_check.setProperty('type', 1)
        self.ch1_check.setEnabled(False)
        self.ch2_check = QtWidgets.QCheckBox('CH2', self)
        self.ch2_check.setProperty('type', 1)
        self.ch2_check.setEnabled(False)
        self.ch3_check = QtWidgets.QCheckBox('CH3', self)
        self.ch3_check.setProperty('type', 1)
        self.ch3_check.setEnabled(False)
        self.ch4_check = QtWidgets.QCheckBox('CH4', self)
        self.ch4_check.setProperty('type', 1)
        self.ch4_check.setEnabled(False)
        self.ch5_check = QtWidgets.QCheckBox('CH5', self)
        self.ch5_check.setProperty('type', 1)
        self.ch5_check.setEnabled(False)
        self.ch6_check = QtWidgets.QCheckBox('CH6', self)
        self.ch6_check.setProperty('type', 1)
        self.ch6_check.setEnabled(False)
        self.ch7_check = QtWidgets.QCheckBox('CH7', self)
        self.ch7_check.setProperty('type', 1)
        self.ch7_check.setEnabled(False)
        self.ch8_check = QtWidgets.QCheckBox('CH8', self)
        self.ch8_check.setProperty('type', 1)
        self.ch8_check.setEnabled(False)

        self.check_group = QtWidgets.QButtonGroup()
        self.check_group.setExclusive(False)
        self.check_group.addButton(self.ch1_check, id=1)
        self.check_group.addButton(self.ch2_check, id=2)
        self.check_group.addButton(self.ch3_check, id=3)
        self.check_group.addButton(self.ch4_check, id=4)
        self.check_group.addButton(self.ch5_check, id=5)
        self.check_group.addButton(self.ch6_check, id=6)
        self.check_group.addButton(self.ch7_check, id=7)
        self.check_group.addButton(self.ch8_check, id=8)

        self.display_all = QtWidgets.QPushButton('Display all', self)
        self.clear_all = QtWidgets.QPushButton('Clear all', self)
        self.display_all.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.clear_all.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

        layout2 = QtWidgets.QGridLayout()

        layout2.addWidget(self.ch1_check, 0, 0)
        layout2.addWidget(self.ch2_check, 1, 0)
        layout2.addWidget(self.ch3_check, 2, 0)
        layout2.addWidget(self.ch4_check, 3, 0)
        layout2.addWidget(self.ch5_check, 0, 1)
        layout2.addWidget(self.ch6_check, 1, 1)
        layout2.addWidget(self.ch7_check, 2, 1)
        layout2.addWidget(self.ch8_check, 3, 1)
        layout2.addWidget(self.display_all, 4, 0)
        layout2.addWidget(self.clear_all, 4, 1)

        self.checks_frame = QtWidgets.QGroupBox('Channels')
        self.checks_frame.setLayout(layout2)

        self.raw = QtWidgets.QRadioButton('RAW', self)
        self.raw.setProperty('type', 1)
        self.raw.setEnabled(False)
        self.filt = QtWidgets.QRadioButton('Filtered', self)
        self.filt.setProperty('type', 1)
        self.filt.setEnabled(False)
        self.rms = QtWidgets.QRadioButton('Averaged RMS Envelope', self)
        self.raw.setProperty('type', 1)
        self.rms.setEnabled(False)
        self.norm = QtWidgets.QRadioButton('Normalized', self)
        self.norm.setProperty('type', 1)
        self.norm.setEnabled(False)

        self.data_group = QtWidgets.QButtonGroup()
        self.data_group.addButton(self.raw, id=0)
        self.data_group.addButton(self.filt, id=1)
        self.data_group.addButton(self.rms, id=2)
        self.data_group.addButton(self.norm, id=3)

        layout6 = QtWidgets.QVBoxLayout()
        layout6.addWidget(self.raw)
        layout6.addWidget(self.filt)
        layout6.addWidget(self.rms)
        layout6.addWidget(self.norm)

        self.type_frame = QtWidgets.QGroupBox('Datatype')
        self.type_frame.setLayout(layout6)

        self.space = QtWidgets.QWidget()

        layout3 = QtWidgets.QGridLayout()

        layout3.addWidget(self.checks_frame, 0, 0)
        layout3.addWidget(self.type_frame, 1, 0)

        config = QtWidgets.QWidget()
        config.setLayout(layout3)

        #  Visualization Panel


        # Model Loss and Accuracy tab
        self.model_acc_loss = FigureCanvas(Figure(figsize=(5, 5)))
        self._mal_fig = self.model_acc_loss.figure
        self._mal_ax = self.model_acc_loss.figure.subplots()
        self._mal_fig.set_tight_layout(True)
        self._mal_ax.set_xlabel('Epochs')
        self._mal_ax.set_ylabel('Accuracy')
        self._mal_ax.set_title('Model Cross-Validation')

        # Adding Twin Axes to plot using dataset_2
        self._mal_ax2 = self._mal_ax.twinx()
        self._mal_ax2.set_ylabel('Loss')

        # Create layout for tsne 3d tab
        layout4 = QtWidgets.QGridLayout()

        self.t_sne3d = gl.GLViewWidget()

        g = gl.GLGridItem()
        self.t_sne3d.addItem(g)

        self.tsne_cbar = FigureCanvas(Figure(figsize=(0.1, 8), facecolor='black'))
        self._tsne_cbar_fig = self.tsne_cbar.figure
        self._tsne_cbar_ax = self.tsne_cbar.figure.subplots()
        # self._tsne_cbar_fig.subplots_adjust(right=0.15)
        self._tsne_cbar_fig.set_tight_layout(True)

        labels = ['O', 'C', 'TO', 'TP', 'BO', 'BC']

        self._tsne_cbar_fig.colorbar(mpl.cm.ScalarMappable(norm=utils.norm, cmap=mpl.cm.Spectral),
             cax=self._tsne_cbar_ax, orientation='vertical')
        
        # Shift ticks to be at 0.5, 1.5, etc
        self._tsne_cbar_ax.yaxis.set(ticks=np.arange(0.5, len(labels)), ticklabels=labels)
        self._tsne_cbar_ax.tick_params(axis='y', colors='ghostwhite')

        layout4.addWidget(self.t_sne3d, 0, 0)
        layout4.addWidget(self.tsne_cbar, 0, 1)
        layout4.setColumnMinimumWidth(0, 300)
        layout4.setColumnMinimumWidth(1, 90)
        layout4.setColumnStretch(0, 1)
        layout4.setColumnStretch(1, 0)

        tsne_tab = QtWidgets.QWidget()
        tsne_tab.setLayout(layout4)

        # Confusion Matrix tab
        self.cm = FigureCanvas(Figure(figsize=(5, 5)))
        self._cm_fig = self.cm.figure
        self._cm_ax = self.cm.figure.subplots()
        self._cm_fig.set_tight_layout(True)

        vis_tabs = QtWidgets.QTabWidget(self)
        vis_tabs.addTab(self.model_acc_loss, "Cross-Validation")
        vis_tabs.addTab(self.cm, "Confusion Matrix")
        vis_tabs.addTab(tsne_tab, "t-SNE 3D")

        # Create Layout Stack
        self.layout_stack.insertWidget(0, config)  # index 0
        self.layout_stack.insertWidget(1, self.table)  # index 1
        self.layout_stack.insertWidget(2, vis_tabs)  # index 2

        # Assemble widgets
        layout5 = QtWidgets.QVBoxLayout()

        layout5.addLayout(self.layout_stack)
        layout5.addWidget(self.options_frame)

        main_frame = QtWidgets.QWidget()
        main_frame.setLayout(layout5)

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter1.addWidget(self.TASK)
        splitter1.addWidget(main_frame)

        splitter2 = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter2.addWidget(channels_frame)
        splitter2.addWidget(splitter1)

        # Add splitters to the final layout
        layout_win = QtWidgets.QHBoxLayout()
        layout_win.addWidget(splitter2)

        self.setLayout(layout_win)
        self.layout_stack.setCurrentIndex(self.options_group.checkedId())  # Make Default the options tab

    def activate_tab(self):
        self.layout_stack.setCurrentIndex(self.options_group.checkedId())


class MainUi(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(MainUi, self).__init__(parent)

        """Create the General page UI."""
        # Title label
        self.title_label = QtWidgets.QLabel('Intention Detection and Task Classification', self)
        id = QtGui.QFontDatabase.addApplicationFont('dev/assets/fonts/TitilliumWeb-Bold.ttf')
        families = QtGui.QFontDatabase.applicationFontFamilies(id)
        self.title_label.setFont(QtGui.QFont(families[0], 48))
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        # Countdown label
        self.countdown_label = QtWidgets.QLabel(self)
        self.countdown_label.setFont(QtGui.QFont(families[0], 24))
        self.countdown_label.setProperty('type', 5)
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)

        # Timer label
        self.timer_label = QtWidgets.QLabel(self)
        self.timer_label.setFont(QtGui.QFont(families[0], 24))
        self.timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)
        self.timer_label.setProperty('type', 5)
        # Detection Label
        self.detection_label = QtWidgets.QLabel(self)
        self.detection_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom)
        self.detection_label.setFont(QtGui.QFont(families[0], 24))
        # Image label
        self.task_img = QtWidgets.QLabel(self)
        self.task_img.setMinimumSize(1,1)
        self.task_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        # Push buttons
        self.collect = QtWidgets.QPushButton("Collect Data", self)
        # self.collect.setProperty('type', 1)
        self.train = QtWidgets.QPushButton("Train Model", self)
        # self.train.setProperty('type', 1)
        self.test = QtWidgets.QPushButton("Test Patient", self)
        # self.test.setProperty('type', 1)
        self.interrupt = QtWidgets.QPushButton("Interrupt", self)
        self.interrupt.setHidden(True)
        self.interrupt.setProperty('type', 3)
        # Display Instructions
        file = open(config.resource_path('dev/assets/text/mainInstructions.txt'))
        text = file.read()
        file.close()
        self.instruct = QtWidgets.QPlainTextEdit(self)
        self.instruct.setPlainText(text)
        self.instruct.setReadOnly(True)

        # Channels
        layout_channels = QtWidgets.QGridLayout()

        self.ch1_main = QtWidgets.QCheckBox('CH1')
        self.ch1_main.setProperty('type', 1)
        self.ch1_main.setEnabled(False)
        self.ch2_main = QtWidgets.QCheckBox('CH2')
        self.ch2_main.setProperty('type', 1)
        self.ch2_main.setEnabled(False)
        self.ch3_main = QtWidgets.QCheckBox('CH3')
        self.ch3_main.setProperty('type', 1)
        self.ch3_main.setEnabled(False)
        self.ch4_main = QtWidgets.QCheckBox('CH4')
        self.ch4_main.setProperty('type', 1)
        self.ch4_main.setEnabled(False)
        self.ch5_main = QtWidgets.QCheckBox('CH5')
        self.ch5_main.setProperty('type', 1)
        self.ch5_main.setEnabled(False)
        self.ch6_main = QtWidgets.QCheckBox('CH6')
        self.ch6_main.setProperty('type', 1)
        self.ch6_main.setEnabled(False)
        self.ch7_main = QtWidgets.QCheckBox('CH7')
        self.ch7_main.setProperty('type', 1)
        self.ch7_main.setEnabled(False)
        self.ch8_main = QtWidgets.QCheckBox('CH8')
        self.ch8_main.setProperty('type', 1)
        self.ch8_main.setEnabled(False)

        self.check_group = QtWidgets.QButtonGroup()
        self.check_group.setExclusive(False)
        self.check_group.addButton(self.ch1_main, id=1)
        self.check_group.addButton(self.ch2_main, id=2)
        self.check_group.addButton(self.ch3_main, id=3)
        self.check_group.addButton(self.ch4_main, id=4)
        self.check_group.addButton(self.ch5_main, id=5)
        self.check_group.addButton(self.ch6_main, id=6)
        self.check_group.addButton(self.ch7_main, id=7)
        self.check_group.addButton(self.ch8_main, id=8)
        self.check_group.idClicked.connect(self.updateChannels)

        layout_channels.addWidget(self.ch1_main, 0, 0)
        layout_channels.addWidget(self.ch2_main, 0, 1)
        layout_channels.addWidget(self.ch3_main, 0, 2)
        layout_channels.addWidget(self.ch4_main, 0, 3)
        layout_channels.addWidget(self.ch5_main, 1, 0)
        layout_channels.addWidget(self.ch6_main, 1, 1)
        layout_channels.addWidget(self.ch7_main, 1, 2)
        layout_channels.addWidget(self.ch8_main, 1, 3)

        self.channels_frame = QtWidgets.QGroupBox('Channels')
        self.channels_frame.setLayout(layout_channels)

        # Status Labels
        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/subject.png'))
        self.subject_icon = QtWidgets.QToolButton()
        self.subject_icon.setIcon(icon)
        self.subject_icon.setAutoRaise(False)
        newTip = "Open an existing subject's folder or create a new one"
        self.subject_icon.setToolTip(newTip)

        self.subject_label = QtWidgets.QLabel('Subject: ')
        self.subject_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.subject_label.setProperty('type', 1)

        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/table.png'))
        self.data_icon = QtWidgets.QToolButton()
        self.data_icon.setIcon(icon)
        self.data_icon.setAutoRaise(False)
        newTip = "Open a dataset"
        self.data_icon.setToolTip(newTip)

        self.data_label = QtWidgets.QLabel('Data: ', self)
        self.data_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.data_label.setProperty('type', 1)

        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/folder-open.png'))
        self.model_icon = QtWidgets.QToolButton()
        self.model_icon.setIcon(icon)
        self.model_icon.setAutoRaise(False)
        newTip = "Open a new model"
        self.model_icon.setToolTip(newTip)

        self.model_label = QtWidgets.QLabel('Model: ', self)
        self.model_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.model_label.setProperty('type', 1)

        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/time.png'))
        self.time_icon = QtWidgets.QToolButton()
        self.time_icon.setIcon(icon)
        self.time_icon.setAutoRaise(False)
        newTip = "Change time interval"
        self.time_icon.setToolTip(newTip)

        self.time_interval_label = QtWidgets.QLabel('Time Interval: ', self)
        self.time_interval_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.time_interval_label.setProperty('type', 1)

        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/data-transfer.png'))
                           
        self.mindrove_icon = QtWidgets.QToolButton()
        self.mindrove_icon.setIcon(icon)
        self.mindrove_icon.setAutoRaise(False)
        newTip = "Check status of Mindrove"
        self.mindrove_icon.setToolTip(newTip)

        self.mindrove_label = QtWidgets.QLabel('MindRove: ', self)
        self.mindrove_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.mindrove_label.setProperty('type', 1)

        icon = QtGui.QIcon(config.resource_path('dev/assets/icons/ethernet.png'))
        self.port_icon = QtWidgets.QToolButton()
        self.port_icon.setIcon(icon)
        self.port_icon.setAutoRaise(False)
        newTip = "Check status of Port"
        self.port_icon.setToolTip(newTip)

        self.port_status_label = QtWidgets.QLabel('Port: ', self)
        self.port_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.port_status_label.setProperty('type', 1)

        #  Status
        self.task_instances_frame = QtWidgets.QGroupBox('Task Instances')
        self.task_instances_frame.setFlat(True)

        task_open_label = QtWidgets.QLabel('Open: ', self)
        task_open_label.setProperty('type', 2)
        task_close_label = QtWidgets.QLabel('Close: ', self)
        task_close_label.setProperty('type', 2)
        task_tripod_open_label = QtWidgets.QLabel('Tripod Open: ', self)
        task_tripod_open_label.setProperty('type', 2)
        task_tripod_pinch_label = QtWidgets.QLabel('Tripod Pinch: ', self)
        task_tripod_pinch_label.setProperty('type', 2)
        task_bottom_open_label = QtWidgets.QLabel('Bottom Open: ', self)
        task_bottom_open_label.setProperty('type', 2)
        task_bottom_close_label = QtWidgets.QLabel('Bottom Close: ', self)
        task_bottom_close_label.setProperty('type', 2)

        self.task_open = QtWidgets.QLabel('0', self)
        self.task_open.setProperty('type', 2)
        self.task_close = QtWidgets.QLabel('0', self)
        self.task_close.setProperty('type', 2)
        self.task_tripod_open = QtWidgets.QLabel('0', self)
        self.task_tripod_open.setProperty('type', 2)
        self.task_tripod_pinch = QtWidgets.QLabel('0', self)
        self.task_tripod_pinch.setProperty('type', 2)
        self.task_bottom_open = QtWidgets.QLabel('0', self)
        self.task_bottom_open.setProperty('type', 2)
        self.task_bottom_close = QtWidgets.QLabel('0', self)
        self.task_bottom_close.setProperty('type', 2)

        tasks_status_layout = QtWidgets.QGridLayout()

        tasks_status_layout.addWidget(task_open_label, 0, 0)
        tasks_status_layout.addWidget(self.task_open, 0, 1)
        tasks_status_layout.addWidget(task_close_label, 1, 0)
        tasks_status_layout.addWidget(self.task_close, 1, 1)
        tasks_status_layout.addWidget(task_tripod_open_label, 0, 2)
        tasks_status_layout.addWidget(self.task_tripod_open, 0, 3)
        tasks_status_layout.addWidget(task_tripod_pinch_label, 1, 2)
        tasks_status_layout.addWidget(self.task_tripod_pinch, 1, 3)
        tasks_status_layout.addWidget(task_bottom_open_label, 0, 4)
        tasks_status_layout.addWidget(self.task_bottom_open, 0, 5)
        tasks_status_layout.addWidget(task_bottom_close_label, 1, 4)
        tasks_status_layout.addWidget(self.task_bottom_close, 1, 5)
        
        tasks_status_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.task_instances_frame.setLayout(tasks_status_layout)

        self.subject = QtWidgets.QLabel('', self)
        self.subject.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.subject.setProperty('type', 2)

        self.data_status = QtWidgets.QLabel('', self)
        self.data_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.data_status.setProperty('type', 2)

        self.model_status = QtWidgets.QLabel('', self)
        self.model_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.model_status.setProperty('type', 2)

        self.time_interval = QtWidgets.QLabel('Active/Inactive: 3', self) # Default timeinterval is 3 seconds for active and inactive interval
        self.time_interval.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.time_interval.setProperty('type', 2)

        self.mindrove_status = QtWidgets.QLabel('', self)
        self.mindrove_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.mindrove_status.setProperty('type', 2)

        self.port_status = QtWidgets.QLabel('Not Connected', self)
        self.port_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.port_status.setProperty('type', 2)

        layout1 = QtWidgets.QHBoxLayout()
        layout2 = QtWidgets.QHBoxLayout()
        layout3 = QtWidgets.QHBoxLayout()
        layout4 = QtWidgets.QHBoxLayout()
        layout5 = QtWidgets.QHBoxLayout()
        layout6 = QtWidgets.QHBoxLayout()

        layout1.addWidget(self.subject_icon)
        layout1.addWidget(self.subject_label)
        layout2.addWidget(self.data_icon)
        layout2.addWidget(self.data_label)
        layout3.addWidget(self.model_icon)
        layout3.addWidget(self.model_label)
        layout4.addWidget(self.time_icon)
        layout4.addWidget(self.time_interval_label)
        layout5.addWidget(self.mindrove_icon)
        layout5.addWidget(self.mindrove_label)
        layout6.addWidget(self.port_icon)
        layout6.addWidget(self.port_status_label)
        
        layout_diagnostics = QtWidgets.QGridLayout()

        layout_diagnostics.addWidget(self.task_instances_frame, 0, 0, 1, 2) # Did not include icon as seen in layout1
        layout_diagnostics.addLayout(layout1, 1, 0)
        layout_diagnostics.addLayout(layout2, 2, 0)
        layout_diagnostics.addLayout(layout3, 3, 0)
        layout_diagnostics.addLayout(layout4, 4, 0)
        layout_diagnostics.addLayout(layout5, 5, 0)
        layout_diagnostics.addLayout(layout6, 6, 0)
        layout_diagnostics.addWidget(self.subject, 1, 1)
        layout_diagnostics.addWidget(self.data_status, 2, 1)
        layout_diagnostics.addWidget(self.model_status, 3, 1)
        layout_diagnostics.addWidget(self.time_interval, 4, 1)
        layout_diagnostics.addWidget(self.mindrove_status, 5, 1)
        layout_diagnostics.addWidget(self.port_status, 6, 1)
        
        layout_diagnostics.setColumnStretch(1, 1)

        self.diagnostic_frame = QtWidgets.QGroupBox('Status')

        self.diagnostic_frame.setLayout(layout_diagnostics)

        self.port_label = QtWidgets.QLabel("Port:")
        # self.port_label.setStyleSheet("font-weight: bold; color: ghostwhite")
        self.port_select = QtWidgets.QComboBox()
        self.detect = QtWidgets.QPushButton("Detect", self)
        self.detect.setProperty('type', 2)
        self.connect_port = QtWidgets.QPushButton("Connect", self)
        self.connect_port.setProperty('type', 2)

        port_layout = QtWidgets.QHBoxLayout()
        port_layout.addWidget(self.port_label, stretch=0)
        port_layout.addWidget(self.port_select, stretch=1)
        port_layout.addWidget(self.detect, stretch=1)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.instruct, stretch=1)
        layout_controls.addWidget(self.channels_frame)
        layout_controls.addWidget(self.diagnostic_frame)
        layout_controls.addLayout(port_layout)
        layout_controls.addWidget(self.connect_port)

        controls_frame = QtWidgets.QWidget()
        controls_frame.setLayout(layout_controls)

        # Create a QGridLayout instance
        layout_main = QtWidgets.QGridLayout()
        layout_main.setColumnStretch(0, 1)
        layout_main.setColumnStretch(1, 1)
        layout_main.setColumnStretch(2, 1)

        self.pb = QtWidgets.QProgressBar()
        self.pb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pb.setHidden(True)

        # Add widgets to the layout
        layout_main.addWidget(self.task_img, 0, 0, 5, 3)
        layout_main.addWidget(self.title_label, 0, 0, 4, 3)
        layout_main.addWidget(self.countdown_label, 0, 0)
        layout_main.addWidget(self.timer_label, 0, 2)
        layout_main.addWidget(self.interrupt, 3, 2)
        layout_main.addWidget(self.detection_label, 4, 2)
        layout_main.addWidget(self.pb, 4, 0, 1, 3)
        layout_main.addWidget(self.collect, 5, 0)
        layout_main.addWidget(self.train, 5, 1)
        layout_main.addWidget(self.test, 5, 2)

        main_frame = QtWidgets.QWidget()
        main_frame.setLayout(layout_main)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(main_frame)
        splitter.addWidget(controls_frame)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter)
        splitter.setStretchFactor(0, 1)
        self.setLayout(layout)

    def updateChannels(self):
        pass
