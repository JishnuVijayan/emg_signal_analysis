# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:52:27 2024

@author: jishn
"""
import serial
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
import emg as e

# Load the trained SVM model
svm_model = load('new_working_random_forest_with_more_data.joblib')

# Initialize the StandardScaler with the same parameters used during training
scaler = StandardScaler()
scaler.fit(np.array([[0]])) # Assuming the training data was scaled with a single feature

# Serial port setup
port = "COM4" # Replace with your Arduino's port
baudrate = 9600
ser = serial.Serial(port, baudrate)
time.sleep(2) # Wait for the serial connection to initialize

# Initialize the EMG filter
emg_filter = e.EMG_filter(sample_frequency=200, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=150, reference_available=False)

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # Assuming the line is in the format "time,value"
            # Rename 'time' to 'read_time' to avoid conflict with the 'time' module
            read_time, value = map(float, line.split(','))
            # Print the raw EMG data
            print(f"Raw EMG Data: {value}")
            # Filter the EMG data
            filtered_value = emg_filter.filter(value)
            # Print the filtered EMG data
            print(f"Filtered EMG Data: {filtered_value}")
            # Reshape the data to be a 2D array for the model
            data = np.array([filtered_value]).reshape(-1, 1)
            # Standardize the data
            data = scaler.transform(data)
            # Predict the class of the new value
            prediction = svm_model.predict(data)
            # Map the prediction back to the original labels
            predicted_gesture = 'Hand is Moving' if prediction[0] == 1 else 'Hand is at Rest'
            print(f"Prediction: {predicted_gesture}")
            # Introduce a 1-second delay
            time.sleep(1)
except KeyboardInterrupt:
    ser.close()

