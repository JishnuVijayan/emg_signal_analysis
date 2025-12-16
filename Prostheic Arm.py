# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:17:46 2024

@author: jishn
"""

import serial
import numpy as np
import pandas as pd
from joblib import load
import emg as e  # Assuming you have an 'emg' module with an EMG_filter class

# Load the trained Random Forest model
model = load("random_forest_new2.joblib")

# Set up serial communication with Arduino
ser = serial.Serial('COM5', 9600)  # Replace 'COMX' with the appropriate port

# Initialize the EMG filter
filter = e.EMG_filter(sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600, reference_available=False)

# Variables to track predictions
consecutive_move_count = 0
consecutive_rest_count = 0
anti_clockwise_done = False
clockwise_done = False

# Function to preprocess data and make prediction
def predict_hand_movement(time_stamp, emg_data):
    # Filter the EMG data
    filtered_emg = filter.filter(emg_data)
    # Create a DataFrame with the same column names as the training data
    data = pd.DataFrame([[time_stamp, filtered_emg]], columns=['time', 'filtered_emg'])
    # Predict hand movement without scaling
    prediction = model.predict(data)
    return time_stamp, filtered_emg, prediction[0]

# Main loop for real-time prediction
try:
    while True:
        # Read data from Arduino
        raw_data = ser.readline().decode('utf-8').rstrip()
        # Ensure raw_data is not empty
        if raw_data:
            # Split the data into time and EMG value
            time_stamp, emg_value = map(float, raw_data.split(','))
            # Make prediction
            time_prediction, filtered_emg, movement_prediction = predict_hand_movement(time_stamp, emg_value)
            # Print raw EMG, filtered EMG, and prediction
            print("Time:", time_stamp, "Raw EMG:", emg_value, "Filtered EMG:", filtered_emg, "Prediction:", movement_prediction)

            if movement_prediction == 1:
                consecutive_move_count += 1
                consecutive_rest_count = 0
            else:
                consecutive_rest_count += 1
                consecutive_move_count = 0

            if not anti_clockwise_done and consecutive_move_count >= 10:
                ser.write(b'ANTI_CLOCKWISE\n')
                anti_clockwise_done = True
                consecutive_move_count = 0  # Reset count after action

            if anti_clockwise_done and not clockwise_done and consecutive_rest_count >= 10:
                ser.write(b'CLOCKWISE\n')
                clockwise_done = True
                consecutive_rest_count = 0  # Reset count after action

except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")
