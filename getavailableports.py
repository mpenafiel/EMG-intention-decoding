import serial
from serial.tools import list_ports
import glob
import sys


# from https://stackoverflow.com/questions/12090503./listing-available-com-ports-with-python


def serial_ports():
    """List the available serial ports by name

    : raises error for unexpected or unknown platforms

    : returns:
    list of available serial ports on system"""

    if sys.platform.startswith('win'):
        port_objs = list(list_ports.comports())
        ports = []
        for i in port_objs:
            ports.append(i.device)
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
