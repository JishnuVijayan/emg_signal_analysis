from serial import Serial
import pandas as pd
import numpy as np
from joblib import load
import emg as e

# Load the trained SVM model
clf = load('svm_model.joblib')

# Establish serial connection with Arduino
ser = Serial('COM5', 9600) # Replace 'COM5' with the appropriate port
ser.flushInput()

# Initialize the EMG filter
filter = e.EMG_filter(sample_frequency=1000, range_=0.5, min_EMG_frequency=25, max_EMG_frequency=450, reference_available=False)

# Create a buffer for EMG data
emg_buffer = []

def calculate_features(emg_values):
    # Check for NaN values and handle them
    if np.isnan(emg_values).any():
        # Option 1: Remove NaN values
        emg_values = emg_values[~np.isnan(emg_values)]
        # Option 2: Impute NaN values with the mean
        # emg_values = np.nan_to_num(emg_values, nan=np.mean(emg_values))
    emg_values = (emg_values - np.min(emg_values)) / (np.max(emg_values) - np.min(emg_values))
    mean = np.mean(emg_values)
    std = np.std(emg_values)
    max_val = np.max(emg_values)
    min_val = np.min(emg_values)
    return [mean, std, max_val, min_val]

# Main loop for real-time prediction
while True:
    if ser.in_waiting > 0:
        data = ser.readline().strip().decode('utf-8')
        emg_data = [float(val) for val in data.split(',')]
        emg_values = emg_data[1:]
        
        # Print raw EMG values
        print("Raw EMG values:", emg_values)
        
        emg_buffer.extend(emg_values)
        
        # Process the buffer in chunks of size 10
        while len(emg_buffer) >= 10:
            window = emg_buffer[:10]
            emg_buffer = emg_buffer[10:] # Remove the processed window from the buffer
            
            # Filter the window
            filtered_emg = [filter.filter(value) for value in window]
            
            # Calculate features for the window
            features = calculate_features(filtered_emg)
            
            # Predict using the trained model
            prediction = clf.predict([features])
            if prediction == 1:
                print("Hand is moving")
            else:
                print("Hand is at rest")
