# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:27:20 2024

@author: jishn
"""

import serial

# Establish connection to the serial port that your Arduino is connected to.
# Replace '/dev/ttyUSB0' with the correct port for your Arduino.
ser = serial.Serial('COM5', 9600)

# Open text file to store the received data
with open("diffdata03.txt", 'w') as text_file:
    while True:
        # Read serial data from Arduino
        data = ser.readline().decode('utf-8').strip()
        if data:
            # Write the received data to the text file
            text_file.write(data + '\n')
            text_file.flush()
            print(f"Data written: {data}")
        else:
            break

# Close the serial connection
ser.close()