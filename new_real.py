import serial
import numpy as np
import pandas as pd
from joblib import load
from collections import deque
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Load the trained KNN model and fitted scaler
knn_model = load("knn_model_with_window_5.joblib")
scaler = load("scaler.joblib")

# Set up serial communication with Arduino
try:
    ser = serial.Serial('COM6', 9600)
    print("Serial connection established.")
except serial.SerialException as e:
    print("Error opening serial port: ", e)
    exit()

# Initialize a deque (double-ended queue) for storing the last 10 EMG values
window = deque(maxlen=10)

def predict_hand_movement(time_stamp, emg_data):
    # Add the raw EMG value to the window
    window.append(emg_data)
    # Ensure the window is full before making a prediction
    if len(window) == 10:
        # Calculate the statistical features over the window
        mav = np.mean(np.abs(window))
        rms = np.sqrt(np.mean(np.square(window)))
        sd = np.std(window)
        # Create a list of features
        features = [mav, rms, sd]
        # Scale the features using the previously fitted scaler
        scaled_features = scaler.transform([features])
        # Predict hand movement
        prediction = knn_model.predict(scaled_features)
        return time_stamp, emg_data, prediction[0]
    else:
        return time_stamp, emg_data, None

# Main loop for real-time prediction
try:
    while True:
        # Read data from Arduino
        raw_data = ser.readline().decode('utf-8').rstrip()
        # Print raw data for debugging
        print("Raw data received: ", raw_data)
        # Ensure raw_data is not empty
        if raw_data:
            # Split the data into time and EMG value
            try:
                time_stamp, emg_value = map(float, raw_data.split(','))
            except ValueError:
                print("Error parsing data: ", raw_data)
                continue
            # Make prediction
            time_prediction, raw_emg, movement_prediction = predict_hand_movement(time_stamp, emg_value)
            # Print raw EMG and prediction
            print("Time:", time_stamp, "Raw EMG:", emg_value, "Prediction:", movement_prediction)
            if movement_prediction is not None:
                if movement_prediction == 0:
                    print("Hand is at Rest")
                elif movement_prediction == 1:
                    print("Hand is Moving Right")
                elif movement_prediction == 2:
                    print("Hand is Moving LEFT")
                elif movement_prediction == 3:
                    print("Hand is Moving UP")
                elif movement_prediction == 4:
                    print("Hand is Moving Down")
except serial.SerialException as e:
    print("Serial communication error: ", e)
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")

