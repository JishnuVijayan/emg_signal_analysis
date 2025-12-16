# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:01:58 2024

@author: jishn
"""

import serial
import numpy as np
import pandas as pd
from joblib import load
import emg as e  # Assuming you have an 'emg' module with an EMG_filter class

# Load the trained KNeighborsClassifier model and scaler
knn, scaler = load("KNN_MODEL_WITH_SCALER.joblib")

# Set up serial communication with Arduino
ser = serial.Serial('COM5', 9600)  # Replace 'COMX' with the appropriate port

# Initialize the EMG filter
filter = e.EMG_filter(
    sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600, reference_available=False
)

def predict_hand_movement(time_stamp, emg_data):
    # Filter the EMG data
    filtered_emg = filter.filter(emg_data)
    
    # Create input data with the same format as training data
    input_data = [[time_stamp, filtered_emg]]
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)  # Input must match training dimensions
    
    # Predict hand movement
    prediction = knn.predict(scaled_data)
    
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
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            if movement_prediction == 0:
                
                print("Muscle is relaxed")
            else:
                print("Muscle is activated")
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")
    


