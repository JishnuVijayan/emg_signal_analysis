# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:03:32 2024

@author: jishn
"""

import serial
import numpy as np
import pandas as pd
from joblib import load
import emg as e
from collections import deque

# Load the trained Random Forest model and feature names
model, feature_names = load("latest_model.joblib")

# Set up serial communication with Arduino
ser = serial.Serial('COM5', 9600)

# Initialize the EMG filter
filter = e.EMG_filter(sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600, reference_available=False)

# Initialize a deque (double-ended queue) for storing the last 10 EMG values
window = deque(maxlen=10)

def predict_hand_movement(time_stamp, emg_data):
    # Filter the EMG data
    filtered_emg = filter.filter(emg_data)
    # Add the filtered EMG value to the window
    window.append(filtered_emg)
    # Ensure the window is full before making a prediction
    if len(window) == 10:
        # Calculate the statistical features over the window
        mean = np.mean(window)
        std = np.std(window)
        min_val = np.min(window)
        max_val = np.max(window)
        median = np.median(window)
        # Create a list of features
        features = [time_stamp, filtered_emg, mean, std, min_val, max_val, median]
        # Repeat the features for each window
        features = features * 11
        # Create a DataFrame with the same column order as the training data
        data = pd.DataFrame([features], columns=feature_names)
        # Predict hand movement
        prediction = model.predict(data)
        return time_stamp, filtered_emg, prediction[0]
    else:
        return time_stamp, filtered_emg, None



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
            if movement_prediction is not None:
                if movement_prediction == 0:
                    print(" Hand is at Rest")
                elif movement_prediction == 1:
                    print("Hand is Moving Right")
                elif movement_prediction == 2:
                    print(" Hand is Moving LEFT")
                elif movement_prediction == 3:
                    print("Hand is Moving UP")
                elif movement_prediction == 4:
                    print("Hand is Moving Down")
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")


