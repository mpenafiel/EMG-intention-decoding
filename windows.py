from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import matplotlib

matplotlib.use('QtAgg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import getavailableports


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class CheckBoxDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Training Parameters')

        Qbtn = QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(Qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.close = QtWidgets.QCheckBox('Close', self)
        self.open = QtWidgets.QCheckBox('Open', self)
        self.tripod = QtWidgets.QCheckBox('Tripod', self)
        self.tripod_open = QtWidgets.QCheckBox('Tripod Open', self)

        self.poses_frame = QtWidgets.QGroupBox('Training Poses')

        layout1 = QtWidgets.QVBoxLayout()
        layout1.addWidget(self.close)
        layout1.addWidget(self.open)
        layout1.addWidget(self.tripod)
        layout1.addWidget(self.tripod_open)

        self.poses_frame.setLayout(layout1)

        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(self.poses_frame)
        layout2.addWidget(self.buttonBox)

        self.setLayout(layout2)

        #  create a dictionary with the associated values
        self.poses = {self.close: 1,
                      self.open: 2,
                      self.tripod: 3,
                      self.tripod_open: 4}


def instruct():
    instructions = '''
        EMG Hand Calibrations

        This is the document that will be displayed on the window. This will provide primary instructions for the 
        patient but will be easy to display with low storage.

        In this paragraph, basic information can be displayed so that users understand the functions of the application.

        DO NOT CLOSE THE WINDOW WHILE DATA IS BEING COLLECTED
        '''
    return instructions


def createMain():
    win = MainUi()
    return win


def createAnalysis():
    win = AnalysisUI()
    return win


def createPatientView():
    win = ViewWin()
    return win


class ViewWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Window")
        # Set the geometry of the window
        self.setGeometry(500, 100, 1200, 800)

        self.pose_img = QtWidgets.QLabel(self)
        self.pose_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
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
        layout.addWidget(self.pose_img)
        layout.addWidget(self.pb)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)


class AnalysisUI(QtWidgets.QWidget):

    def __init__(self):
        super(AnalysisUI, self).__init__()

        layout_win = QtWidgets.QGridLayout()

        self.layout_stack = QtWidgets.QStackedLayout()  # referenced in method

        self.CH0 = pg.PlotWidget(name='CH0')
        self.CH0.setLabel('left', 'Potential', units='V')
        self.CH0.setLabel('bottom', 'Time', units='s')
        self.CH0.setTitle("CH0")

        self.CH1 = pg.PlotWidget(name="CH1")
        self.CH1.setLabel('left', 'Potential', units='V')
        self.CH1.setLabel('bottom', 'Time', units='s')
        self.CH1.setTitle("CH1")

        self.CH2 = pg.PlotWidget(name='CH2')
        self.CH2.setLabel('left', 'Potential', units='V')
        self.CH2.setLabel('bottom', 'Time', units='s')
        self.CH2.setTitle("CH2")

        self.CH3 = pg.PlotWidget(name='CH3')
        self.CH3.setLabel('left', 'Potential', units='V')
        self.CH3.setLabel('bottom', 'Time', units='s')
        self.CH3.setTitle("CH3")

        self.CH4 = pg.PlotWidget(name="CH4")
        self.CH4.setLabel('left', 'Potential', units='V')
        self.CH4.setLabel('bottom', 'Time', units='s')
        self.CH4.setTitle("CH4")

        self.CH5 = pg.PlotWidget(name='CH5')
        self.CH5.setLabel('left', 'Potential', units='V')
        self.CH5.setLabel('bottom', 'Time', units='s')
        self.CH5.setTitle("CH5")

        self.CH6 = pg.PlotWidget(name='CH6')
        self.CH6.setLabel('left', 'Potential', units='V')
        self.CH6.setLabel('bottom', 'Time', units='s')
        self.CH6.setTitle("CH6")

        self.CH7 = pg.PlotWidget(name='CH7')
        self.CH7.setLabel('left', 'Potential', units='V')
        self.CH7.setLabel('bottom', 'Time', units='s')
        self.CH7.setTitle("CH7")

        self.POSE = pg.PlotWidget(name='Pose')
        self.POSE.setLabel('bottom', 'Time', units='s')
        position_labels = [
            # Generate a list of tuples (x_value, x_label)
            (0, 'Rest'), (1, 'Close'), (2, 'Open')
        ]
        ax = self.POSE.getAxis('left')
        # Pass the list in, *in* a list.
        ax.setTicks([position_labels])
        self.POSE.setTitle("Pose")

        self.table = QtWidgets.QTableView()

        self.table_button = QtWidgets.QRadioButton('Table', self)
        self.table_button.setCheckable(True)
        self.options_button = QtWidgets.QRadioButton('Configurations', self)
        self.options_button.setCheckable(True)
        self.stat_button = QtWidgets.QRadioButton('Statistics', self)
        self.stat_button.setCheckable(True)

        self.options_group = QtWidgets.QButtonGroup()
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

        #  Options layout
        self.ch0_check = QtWidgets.QCheckBox('CH0', self)
        self.ch0_check.setEnabled(False)
        self.ch1_check = QtWidgets.QCheckBox('CH1', self)
        self.ch1_check.setEnabled(False)
        self.ch2_check = QtWidgets.QCheckBox('CH2', self)
        self.ch2_check.setEnabled(False)
        self.ch3_check = QtWidgets.QCheckBox('CH3', self)
        self.ch3_check.setEnabled(False)
        self.ch4_check = QtWidgets.QCheckBox('CH4', self)
        self.ch4_check.setEnabled(False)
        self.ch5_check = QtWidgets.QCheckBox('CH5', self)
        self.ch5_check.setEnabled(False)
        self.ch6_check = QtWidgets.QCheckBox('CH6', self)
        self.ch6_check.setEnabled(False)
        self.ch7_check = QtWidgets.QCheckBox('CH7', self)
        self.ch7_check.setEnabled(False)

        self.check_group = QtWidgets.QButtonGroup()
        self.check_group.setExclusive(False)
        self.check_group.addButton(self.ch0_check, id=0)
        self.check_group.addButton(self.ch1_check, id=1)
        self.check_group.addButton(self.ch2_check, id=2)
        self.check_group.addButton(self.ch3_check, id=3)
        self.check_group.addButton(self.ch4_check, id=4)
        self.check_group.addButton(self.ch5_check, id=5)
        self.check_group.addButton(self.ch6_check, id=6)
        self.check_group.addButton(self.ch7_check, id=7)

        self.display_all = QtWidgets.QPushButton('Display all', self)
        self.clear_all = QtWidgets.QPushButton('Clear all', self)
        self.set_channels = QtWidgets.QPushButton('Set Channels', self)
        self.display_all.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.clear_all.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.set_channels.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)

        layout2 = QtWidgets.QGridLayout()

        layout2.addWidget(self.ch0_check, 0, 0)
        layout2.addWidget(self.ch1_check, 1, 0)
        layout2.addWidget(self.ch2_check, 2, 0)
        layout2.addWidget(self.ch3_check, 3, 0)
        layout2.addWidget(self.ch4_check, 0, 1)
        layout2.addWidget(self.ch5_check, 1, 1)
        layout2.addWidget(self.ch6_check, 2, 1)
        layout2.addWidget(self.ch7_check, 3, 1)
        layout2.addWidget(self.display_all, 4, 0)
        layout2.addWidget(self.clear_all, 4, 1)
        layout2.addWidget(self.set_channels, 5, 0, 1, 2)
        layout2.addWidget(self.set_channels, 6, 0, 1, 2)

        layout2.setRowStretch(4, 1)

        self.checks_frame = QtWidgets.QGroupBox('Channels')
        self.checks_frame.setLayout(layout2)

        self.raw = QtWidgets.QRadioButton('RAW', self)
        self.raw.setEnabled(False)
        self.rect = QtWidgets.QRadioButton('Rectified', self)
        self.rect.setEnabled(False)
        self.norm = QtWidgets.QRadioButton('Normalized', self)
        self.norm.setEnabled(False)

        self.data_group = QtWidgets.QButtonGroup()
        self.data_group.addButton(self.raw, id=0)
        self.data_group.addButton(self.rect, id=1)

        self.space = QtWidgets.QWidget()

        layout3 = QtWidgets.QGridLayout()

        layout3.addWidget(self.checks_frame, 0, 0)
        layout3.addWidget(self.raw, 1, 0)
        layout3.addWidget(self.rect, 2, 0)
        layout3.addWidget(self.norm, 3, 0)
        layout3.addWidget(self.space, 0, 1, 3, 1)

        config = QtWidgets.QWidget()
        config.setLayout(layout3)

        #  Statistics layout

        layout4 = QtWidgets.QVBoxLayout()
        self.cm = MplCanvas(self, width=5, height=4, dpi=100)

        # layout4.addWidget(self.cm)

        stats = QtWidgets.QWidget()
        stats.setLayout(layout4)

        self.layout_stack.insertWidget(0, config)  # index 0
        self.layout_stack.insertWidget(1, self.table)  # index 1
        self.layout_stack.insertWidget(2, stats)  # index 2

        layout9 = QtWidgets.QVBoxLayout()

        layout9.addLayout(self.layout_stack)
        layout9.addWidget(self.options_frame)

        # Add widgets to the layout
        layout_win.addWidget(self.CH0, 0, 0)
        layout_win.addWidget(self.CH1, 0, 1)
        layout_win.addWidget(self.CH2, 1, 0)
        layout_win.addWidget(self.CH3, 1, 1)
        layout_win.addWidget(self.CH4, 2, 0)
        layout_win.addWidget(self.CH5, 2, 1)
        layout_win.addWidget(self.CH6, 3, 0)
        layout_win.addWidget(self.CH7, 3, 1)
        layout_win.addWidget(self.POSE, 0, 2)
        layout_win.addLayout(layout9, 1, 2, 3, 1)

        self.setLayout(layout_win)
        self.layout_stack.setCurrentIndex(self.options_group.checkedId())  # Make Default the options tab

    def activate_tab(self):
        self.layout_stack.setCurrentIndex(self.options_group.checkedId())


class MainUi(QtWidgets.QWidget):

    def __init__(self):
        super(MainUi, self).__init__()

        """Create the General page UI."""
        # Countdown label
        self.countdown_label = QtWidgets.QLabel(self)
        self.countdown_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        countdown_font = QtGui.QFont('Ariel', 30)
        countdown_font.setBold(True)
        self.countdown_label.setFont(countdown_font)
        # Timer label
        self.timer_label = QtWidgets.QLabel(self)
        self.timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)
        timer_font = QtGui.QFont('Ariel', 30)
        timer_font.setBold(True)
        self.timer_label.setFont(timer_font)
        # Detection Label
        self.detection_label = QtWidgets.QLabel(self)
        self.detection_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom)
        detection_font = QtGui.QFont('Ariel', 24)
        detection_font.setBold(True)
        self.detection_label.setFont(detection_font)
        # Image label
        self.pose_img = QtWidgets.QLabel(self)
        self.pose_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        # Push buttons
        self.collect = QtWidgets.QPushButton("COLLECT DATA", self)
        self.train = QtWidgets.QPushButton("TRAIN MODEL", self)
        self.test = QtWidgets.QPushButton("TEST PATIENT", self)
        # Display Instructions
        file = open('Intention Detection.txt')
        text = file.read()
        file.close()
        self.instruct = QtWidgets.QPlainTextEdit(self)
        self.instruct.setPlainText(text)
        self.instruct.setReadOnly(True)

        # Channels
        layout_channels = QtWidgets.QGridLayout()

        self.update_channels = QtWidgets.QPushButton(self)
        new_tip = 'Update the available channels to train model on'
        self.update_channels.setToolTip(new_tip)

        self.ch0_main = QtWidgets.QLabel('CH0')
        self.ch0_main.setStyleSheet("font-weight: bold; color: gray")
        self.ch1_main = QtWidgets.QLabel('CH1')
        self.ch1_main.setStyleSheet("font-weight: bold; color: gray")
        self.ch2_main = QtWidgets.QLabel('CH2')
        self.ch2_main.setStyleSheet("font-weight: bold; color: gray")
        self.ch3_main = QtWidgets.QLabel('CH3')
        self.ch3_main.setStyleSheet("font-weight: bold; color: gray")
        self.ch4_main = QtWidgets.QLabel('CH4')
        self.ch4_main.setStyleSheet("font-weight: bold; color: gray")
        self.ch5_main = QtWidgets.QLabel('CH5')
        self.ch5_main.setStyleSheet("font-weight: bold; color: gray")
        self.ch6_main = QtWidgets.QLabel('CH6')
        self.ch6_main.setStyleSheet("font-weight: bold; color: gray")
        self.ch7_main = QtWidgets.QLabel('CH7')
        self.ch7_main.setStyleSheet("font-weight: bold; color: gray")

        layout_channels.addWidget(self.update_channels, 0, 0, 2, 1)
        layout_channels.addWidget(self.ch0_main, 0, 1)
        layout_channels.addWidget(self.ch1_main, 0, 2)
        layout_channels.addWidget(self.ch2_main, 0, 3)
        layout_channels.addWidget(self.ch3_main, 0, 4)
        layout_channels.addWidget(self.ch4_main, 1, 1)
        layout_channels.addWidget(self.ch5_main, 1, 2)
        layout_channels.addWidget(self.ch6_main, 1, 3)
        layout_channels.addWidget(self.ch7_main, 1, 4)

        self.channels_frame = QtWidgets.QGroupBox('Channels')
        self.channels_frame.setStyleSheet("font-weight: bold; color: black")
        self.channels_frame.setLayout(layout_channels)

        # Diagnostic
        icon = QtGui.QIcon('Icons/list.png')
        self.prompt_icon = QtWidgets.QToolButton()
        self.prompt_icon.setIcon(icon)
        self.prompt_icon.setAutoRaise(False)
        newTip = "Create a new prompt"
        self.prompt_icon.setToolTip(newTip)

        self.prompt_label = QtWidgets.QLabel('Prompt: ', self)
        self.prompt_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.prompt_label.setStyleSheet("font-weight: bold; color: black")

        icon = QtGui.QIcon('Icons/table.png')
        self.data_icon = QtWidgets.QToolButton()
        self.data_icon.setIcon(icon)
        self.data_icon.setAutoRaise(False)
        newTip = "Open a dataset"
        self.data_icon.setToolTip(newTip)

        self.data_label = QtWidgets.QLabel('Data: ', self)
        self.data_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.data_label.setStyleSheet("font-weight: bold; color: black")

        icon = QtGui.QIcon('Icons/folder-open.png')
        self.model_icon = QtWidgets.QToolButton()
        self.model_icon.setIcon(icon)
        self.model_icon.setAutoRaise(False)
        newTip = "Open a new model"
        self.model_icon.setToolTip(newTip)

        self.model_label = QtWidgets.QLabel('Model: ', self)
        self.model_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.model_label.setStyleSheet("font-weight: bold; color: black")

        icon = QtGui.QIcon('Icons/data-transfer.png')
        self.mindrove_icon = QtWidgets.QToolButton()
        self.mindrove_icon.setIcon(icon)
        self.mindrove_icon.setAutoRaise(False)
        newTip = "Check status of Mindrove"
        self.mindrove_icon.setToolTip(newTip)

        self.mindrove_label = QtWidgets.QLabel('MindRove: ', self)
        self.mindrove_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.mindrove_label.setStyleSheet("font-weight: bold; color: black")

        icon = QtGui.QIcon('Icons/ethernet.png')
        self.port_icon = QtWidgets.QToolButton()
        self.port_icon.setIcon(icon)
        self.port_icon.setAutoRaise(False)
        newTip = "Check status of Port"
        self.port_icon.setToolTip(newTip)

        self.port_status_label = QtWidgets.QLabel('Port: ', self)
        self.port_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.port_status_label.setStyleSheet("font-weight: bold; color: black")

        #  Status
        self.prompt_status = QtWidgets.QLabel('', self)
        self.prompt_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.prompt_status.setStyleSheet("font-weight: bold; color: black")

        self.data_status = QtWidgets.QLabel('', self)
        self.data_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.data_status.setStyleSheet("font-weight: bold; color: black")

        self.model_status = QtWidgets.QLabel('', self)
        self.model_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.model_status.setStyleSheet("font-weight: bold; color: black")

        self.mindrove_status = QtWidgets.QLabel('', self)
        self.mindrove_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.mindrove_status.setStyleSheet("font-weight: bold; color: black")

        self.port_status = QtWidgets.QLabel('Not Connected', self)
        self.port_status.setStyleSheet("font-weight: bold; color: black")
        self.port_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        layout1 = QtWidgets.QHBoxLayout()
        layout2 = QtWidgets.QHBoxLayout()
        layout3 = QtWidgets.QHBoxLayout()
        layout4 = QtWidgets.QHBoxLayout()
        layout5 = QtWidgets.QHBoxLayout()

        layout1.addWidget(self.prompt_icon)
        layout1.addWidget(self.prompt_label)
        layout2.addWidget(self.data_icon)
        layout2.addWidget(self.data_label)
        layout3.addWidget(self.model_icon)
        layout3.addWidget(self.model_label)
        layout4.addWidget(self.mindrove_icon)
        layout4.addWidget(self.mindrove_label)
        layout5.addWidget(self.port_icon)
        layout5.addWidget(self.port_status_label)

        layout6 = QtWidgets.QVBoxLayout()

        layout6.addLayout(layout1)
        layout6.addLayout(layout2)
        layout6.addLayout(layout3)
        layout6.addLayout(layout4)
        layout6.addLayout(layout5)

        layout7 = QtWidgets.QVBoxLayout()

        layout7.addWidget(self.prompt_status)
        layout7.addWidget(self.data_status)
        layout7.addWidget(self.model_status)
        layout7.addWidget(self.mindrove_status)
        layout7.addWidget(self.port_status)

        layout_diagnostics = QtWidgets.QHBoxLayout()
        layout_diagnostics.setStretch(1, 1)

        layout_diagnostics.addLayout(layout6)
        layout_diagnostics.addLayout(layout7)

        self.diagnostic_frame = QtWidgets.QGroupBox('Status')
        self.diagnostic_frame.setStyleSheet("font-weight: bold; color: black")
        self.diagnostic_frame.setLayout(layout_diagnostics)

        self.port_label = QtWidgets.QLabel("Port:")
        self.port_label.setStyleSheet("font-weight: bold; color: black")
        self.port_select = QtWidgets.QComboBox()
        self.detect = QtWidgets.QPushButton("Detect", self)
        self.connect = QtWidgets.QPushButton("Connect", self)

        port_layout = QtWidgets.QHBoxLayout()
        port_layout.addWidget(self.port_label, stretch=0)
        port_layout.addWidget(self.port_select, stretch=1)
        port_layout.addWidget(self.detect, stretch=1)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.instruct, stretch=1)
        layout_controls.addWidget(self.channels_frame)
        layout_controls.addWidget(self.diagnostic_frame)
        layout_controls.addLayout(port_layout)

        # Create a QGridLayout instance
        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        self.pb = QtWidgets.QProgressBar()
        self.pb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pb.setHidden(True)

        # Add widgets to the layout
        layout.addWidget(self.pose_img, 0, 0, 5, 3)
        layout.addWidget(self.countdown_label, 0, 0)
        layout.addWidget(self.timer_label, 0, 2)
        layout.addWidget(self.detection_label, 4, 2)
        layout.addWidget(self.pb, 4, 0, 1, 3)
        layout.addWidget(self.collect, 5, 0)
        layout.addWidget(self.train, 5, 1)
        layout.addWidget(self.test, 5, 2)
        layout.addLayout(layout_controls, 0, 3, 5, 1)
        layout.addWidget(self.connect, 5, 3)

        self.setLayout(layout)


class FeedbackWin(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Send Feedback')

        self.send_btn = QtWidgets.QPushButton('Send', self)
        self.cancel_btn = QtWidgets.QPushButton('Cancel', self)

        self.send_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        layout_button = QtWidgets.QHBoxLayout()
        layout_button.addWidget(self.send_btn)
        layout_button.addWidget(self.cancel_btn)

        layout_button.setStretch(0, 0)
        layout_button.setStretch(1, 0)

        self.name_label = QtWidgets.QLabel('Name: ', self)
        self.name = QtWidgets.QLineEdit(self)

        self.fb_label = QtWidgets.QLabel('Feedback: ', self)
        self.feedback = QtWidgets.QPlainTextEdit(self)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.name_label, 0, 0)
        layout.addWidget(self.name, 0, 1, 1, 2)
        layout.addWidget(self.fb_label, 1, 0)
        layout.addWidget(self.feedback, 2, 0, 2, 3)

        layout_stack = QtWidgets.QVBoxLayout()

        layout_stack.addLayout(layout)
        layout_stack.addLayout(layout_button)

        layout.setRowStretch(2, 1)

        self.setLayout(layout_stack)

