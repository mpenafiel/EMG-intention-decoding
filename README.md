# EMG Task Classification and Intention Detection
**Version**: 0.5.2

### Overview
The goal of this project is to implement a system that leverages [A Framework of Temporal-Spatial Descriptors-Based Feature Extraction proposed by Rami Kushabi](https://ieeexplore.ieee.org/document/7886279). The framework is light enough to be deployed in real-time decoding, and processes EMG recorded from the low-end, wearable MindRove armband, which has 8 channels and a 500Hz sampling rate.

### Usage
Collect and process raw emg data from the MindRove armband, which includes filtering and detrending. Feature extraction is performed and densely connected network is trained for the EMG profile from a single patient. Patient training occurs when new EMG data is recorded and decoded to determine the intention. The decoded task is relayed via serial port to actuate the Hand of Hope to assist the patient complete the task. This protocol is part of a positive feedback loop where a patient is assisted for intentionaly performing the task.

### List of Files
* `main.py`: code for intializing application and accessing internal functions from the GUI
* `windows.py`: code for managing GUI components
* `utils.py`: code for any funcions that are not a child of the deployed application, but need to be accessed by it
* `model_utils.py`: code containing the parameters of the model and feature extraction functions
* `requirements.txt`: file containing list of all modules and their dependencies.
* `main.spec`: spec file, which can be used to deploy the system as a stand alone executable.