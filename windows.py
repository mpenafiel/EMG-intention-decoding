from PyQt6 import QtWidgets
import pyqtgraph as pg


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
        self.setStyleSheet("background-color: white;")
        # Set the geometry of the window
        self.setGeometry(500, 100, 1200, 800)

        self.image = QtWidgets.QLabel(self)
        self.setCentralWidget(self.image)


class AnalysisUI(QtWidgets.QWidget):

    def __init__(self):
        super(AnalysisUI, self).__init__()

        self.CH0 = pg.PlotWidget(name='CH0')
        self.CH0.setLabel('left', 'Magnitude', units='M')
        self.CH0.setLabel('bottom', 'Time', units='s')
        self.CH0.setTitle("CH0")
        # self.CH0.setXRange(0, 10)

        self.CH1 = pg.PlotWidget(name="CH1")
        self.CH1.setLabel('left', 'Magnitude', units='M')
        self.CH1.setLabel('bottom', 'Time', units='s')
        self.CH1.setTitle("CH1")
        # self.CH1.setXRange(0, 10)

        self.CH2 = pg.PlotWidget(name='CH2')
        self.CH2.setLabel('left', 'Magnitude', units='M')
        self.CH2.setLabel('bottom', 'Time', units='s')
        self.CH2.setTitle("CH2")
        # self.CH2.setXRange(0, 10)

        self.CH3 = pg.PlotWidget(name='CH3')
        self.CH3.setLabel('left', 'Magnitude', units='M')
        self.CH3.setLabel('bottom', 'Time', units='s')
        self.CH3.setTitle("CH3")
        # self.CH3.setXRange(0, 10)

        self.CH4 = pg.PlotWidget(name="CH4")
        self.CH4.setLabel('left', 'Magnitude', units='M')
        self.CH4.setLabel('bottom', 'Time', units='s')
        self.CH4.setTitle("CH4")
        # self.CH4.setXRange(0, 10)

        self.CH5 = pg.PlotWidget(name='CH5')
        self.CH5.setLabel('left', 'Magnitude', units='M')
        self.CH5.setLabel('bottom', 'Time', units='s')
        self.CH5.setTitle("CH5")
        # self.CH5.setXRange(0, 10)

        self.CH6 = pg.PlotWidget(name='CH6')
        self.CH6.setLabel('left', 'Magnitude', units='M')
        self.CH6.setLabel('bottom', 'Time', units='s')
        self.CH6.setTitle("CH6")
        # self.CH6.setXRange(0, 10)

        self.CH7 = pg.PlotWidget(name='CH7')
        self.CH7.setLabel('left', 'Magnitude', units='M')
        self.CH7.setLabel('bottom', 'Time', units='s')
        self.CH7.setTitle("CH7")
        # self.CH7.setXRange(0, 10)

        self.POSE = pg.PlotWidget(name='CH7')
        self.POSE.setLabel('bottom', 'Time', units='s')
        position_labels = [
            # Generate a list of tuples (x_value, x_label)
            (0, 'Rest'), (1, 'Open'), (2, 'Closed')
        ]
        ax = self.POSE.getAxis('left')
        # Pass the list in, *in* a list.
        ax.setTicks([position_labels])
        self.POSE.setTitle("Pose")

        self.table = QtWidgets.QTableView()

        layout = QtWidgets.QGridLayout()

        # Add widgets to the layout
        layout.addWidget(self.CH0, 0, 0)
        layout.addWidget(self.CH1, 0, 1)
        layout.addWidget(self.CH2, 1, 0)
        layout.addWidget(self.CH3, 1, 1)
        layout.addWidget(self.CH4, 2, 0)
        layout.addWidget(self.CH5, 2, 1)
        layout.addWidget(self.CH6, 3, 0)
        layout.addWidget(self.CH7, 3, 1)
        layout.addWidget(self.POSE, 0, 2)
        layout.addWidget(self.table, 1, 2, 3, 1)

        self.setLayout(layout)

        self.p0 = self.CH0.plot()
        self.p1 = self.CH1.plot()
        self.p2 = self.CH2.plot()
        self.p3 = self.CH3.plot()
        self.p4 = self.CH4.plot()
        self.p5 = self.CH5.plot()
        self.p6 = self.CH6.plot()
        self.p7 = self.CH7.plot()


class MainUi(QtWidgets.QWidget):

    def __init__(self):
        super(MainUi, self).__init__()

        """Create the General page UI."""
        # Image label
        self.image = QtWidgets.QLabel(self)
        # Push buttons
        self.collect = QtWidgets.QPushButton("COLLECT DATA", self)
        self.train = QtWidgets.QPushButton("TRAIN DATA", self)
        self.test = QtWidgets.QPushButton("TEST DATA", self)
        # Display Instructions
        self.instruct = QtWidgets.QPlainTextEdit(self)
        text = instruct()
        self.instruct.setPlainText(text)
        self.instruct.setReadOnly(True)
        # Diagnostic
        self.prompt_label = QtWidgets.QLabel('Prompt: ', self)
        self.prompt_label.setStyleSheet("font-weight: bold; color: black")
        self.data_label = QtWidgets.QLabel('Data: ', self)
        self.data_label.setStyleSheet("font-weight: bold; color: black")
        self.model_label = QtWidgets.QLabel('Model: ', self)
        self.model_label.setStyleSheet("font-weight: bold; color: black")
        self.mindrove_label = QtWidgets.QLabel('MindRove Status: ', self)
        self.mindrove_label.setStyleSheet("font-weight: bold; color: black")
        self.daq_label = QtWidgets.QLabel('DAQ Status: ', self)
        self.daq_label.setStyleSheet("font-weight: bold; color: black")

        self.prompt_status = QtWidgets.QLabel('', self)
        self.prompt_status.setStyleSheet("font-weight: bold; color: black")
        self.data_status = QtWidgets.QLabel('', self)
        self.data_status.setStyleSheet("font-weight: bold; color: black")
        self.model_status = QtWidgets.QLabel('', self)
        self.model_status.setStyleSheet("font-weight: bold; color: black")
        self.mindrove_status = QtWidgets.QLabel('', self)
        self.mindrove_status.setStyleSheet("font-weight: bold; color: black")
        self.daq_status = QtWidgets.QLabel('', self)
        self.daq_status.setStyleSheet("font-weight: bold; color: black")

        layout_diagnostics = QtWidgets.QGridLayout()
        layout_diagnostics.setColumnStretch(1, 1)

        layout_diagnostics.addWidget(self.prompt_label, 0, 0)
        layout_diagnostics.addWidget(self.prompt_status, 0, 1)
        layout_diagnostics.addWidget(self.data_label, 1, 0)
        layout_diagnostics.addWidget(self.data_status, 1, 1)
        layout_diagnostics.addWidget(self.model_label, 3, 0)
        layout_diagnostics.addWidget(self.model_status, 3, 1)
        layout_diagnostics.addWidget(self.mindrove_label, 4, 0)
        layout_diagnostics.addWidget(self.mindrove_status, 4, 1)
        layout_diagnostics.addWidget(self.daq_label, 5, 0)
        layout_diagnostics.addWidget(self.daq_status, 5, 1)

        self.diagnostic_frame = QtWidgets.QGroupBox('Diagnostics')
        self.diagnostic_frame.setStyleSheet("font-weight: bold; color: black")
        self.diagnostic_frame.setLayout(layout_diagnostics)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.instruct, stretch=1)
        layout_controls.addWidget(self.diagnostic_frame)

        # Create a QGridLayout instance
        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        # Add widgets to the layout
        layout.addWidget(self.image, 0, 0, 5, 3)
        layout.addWidget(self.collect, 5, 0)
        layout.addWidget(self.train, 5, 1)
        layout.addWidget(self.test, 5, 2)
        layout.addLayout(layout_controls, 0, 3, 5, 1)

        self.setLayout(layout)