# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:01:58 2024

@author: jishn
"""
"""
import serial
import numpy as np
from joblib import load
import pandas as pd
import signal_utilities as su
import emg as e

# Load the trained model
model = load('random_forestnew.joblib')

# Instantiate the EMG_filter class
filter = e.EMG_filter(sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600, reference_available=False)

# Open the serial port
ser = serial.Serial('COM5', 9600)  # replace '/dev/ttyACM0' with the port where your Arduino is connected

while True:
    try:
        # Read the next line from the serial port
        line = ser.readline().decode('utf-8').strip()
        time, raw_emg = map(int, line.split(','))

        # Filter the raw EMG signal
        filtered_emg = filter.filter(raw_emg)

        # Create a DataFrame from the filtered EMG data
        df = pd.DataFrame(data=[filtered_emg], columns=['emg'])

        # Use the trained model to predict the gesture
        prediction = model.predict(df)

        # Print the prediction
        if prediction == 0:
            print('Hand is at Rest')
        else:
            print('Hand is Moving')

    except KeyboardInterrupt:
        # Exit the loop when Ctrl+C is pressed
        break
    except Exception as e:
        # Print any other exceptions and continue
        print(e)
        continue

"""
"""
import serial
import numpy as np
from joblib import load
import pandas as pd
import signal_utilities as su
import emg as e

# Load the trained model and the scaler
model = load('random_forestnew.joblib')
scaler = load('scaler.joblib')

# Instantiate the EMG_filter class
filter = e.EMG_filter(sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600, reference_available=False)

# Open the serial port
ser = serial.Serial('COM5', 9600)  # replace '/dev/ttyACM0' with the port where your Arduino is connected

while True:
    try:
        # Read the next line from the serial port
        line = ser.readline().decode('utf-8').strip()
        time, raw_emg = map(int, line.split(','))

        # Filter the raw EMG signal
        filtered_emg = filter.filter(raw_emg)

        # Create a DataFrame from the filtered EMG data
        df = pd.DataFrame(data=[filtered_emg], columns=['emg'])

        # Scale the data
        df_scaled = scaler.transform(df)

        # Use the trained model to predict the gesture
        prediction = model.predict(df_scaled)

        # Print the prediction
        if prediction == 0:
            print('Hand is at Rest')
        else:
            print('Hand is Moving')

    except KeyboardInterrupt:
        # Exit the loop when Ctrl+C is pressed
        break
    except Exception as e:
        # Print any other exceptions and continue
        print(e)
        continue

"""
#chatgpt

import serial
import numpy as np
import pandas as pd
from joblib import load
import emg as e # Assuming you have an 'emg' module with an EMG_filter class

# Load the trained Random Forest model
model = load("random_forest_new.joblib")

# Set up serial communication with Arduino
ser = serial.Serial('COM5', 9600) # Replace 'COMX' with the appropriate port

# Initialize the EMG filter
filter = e.EMG_filter(sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600, reference_available=False)

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
            #print("Time:", time_stamp, "Raw EMG:", emg_value, "Filtered EMG:", filtered_emg, "Prediction:", movement_prediction)
            if movement_prediction == 0:
                print(" Hand is at rest")
            else:
                print(" Hand is moving")
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")
