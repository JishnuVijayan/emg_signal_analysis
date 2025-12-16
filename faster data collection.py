# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:54:33 2024

@author: jishn
"""

import serial
import time

# Establish connection to the serial port that your Arduino is connected to.
# Replace 'COM3' with the correct port for your Arduino.
ser = serial.Serial('COM3', 9600)

# Open text file to store the received data
with open("updated jishnu.txt", 'w') as text_file:
    start_time = time.time()
    while True:
        # Read serial data from Arduino
        data = ser.readline().decode('utf-8').strip()
        if data:
            # Write the received data to the text file
            text_file.write(data + '\n')
            text_file.flush()
            print(f"Data written: {data}")
        # Stop collecting data after a certain period (e.g., 10 seconds)
        
            

# Close the serial connection
ser.close()
