# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:39:28 2025

@author: jishn
"""

import time
import numpy as np
import pandas as pd
from collections import deque
from joblib import load
import emg as e  # Assuming you have the EMG filter class
import warnings
warnings.filterwarnings("ignore")


# Load the trained ML model and scaler
model = load("SVM_model_with_window_5.joblib")
scaler = load("scaler.joblib")

# Initialize EMG filter
sample_frequency = 100  # Adjust as necessary
range_ = 0.1
min_EMG_frequency = 25
max_EMG_frequency = 600
reference_available = False
emg_filter = e.EMG_filter(sample_frequency, range_, min_EMG_frequency, max_EMG_frequency, reference_available)

# Generate values with filtering and prediction
window_size = 5  # Window size for feature extraction
window = deque(maxlen=window_size)
values = []

# Generate increasing sequence
start_value, max_value = 150, 1024
num_steps = 25
increase_values = np.linspace(start_value, max_value, num_steps, dtype=int)
decrease_values = np.linspace(max_value, start_value, num_steps, dtype=int)

# Hold values for 10 repeats
hold_min = [start_value] * 10
hold_max = [max_value] * 10

# Full cycle sequence
sequence = list(hold_min) + list(increase_values) + list(hold_max) + list(decrease_values)

# Run the sequence continuously
while True:
    for value in sequence:
        # Apply EMG filtering
        filtered_value = emg_filter.filter(value)
        
        # Append to window
        window.append(filtered_value)
        
        # Extract features when the window is full
        if len(window) == window_size:
            mav = np.mean(np.abs(window))
            rms = np.sqrt(np.mean(np.square(window)))
            sd = np.std(window)
            features = [mav, rms, sd]
            
            # Scale features
            scaled_features = scaler.transform([features])
            
            # Predict muscle state
            prediction = model.predict(scaled_features)[0]
            
            # Print prediction
            print(f"Raw EMG: {value}, Filtered: {filtered_value}, Prediction: {'Active' if prediction == 1 else 'Rest'}")
        
        # Delay
        time.sleep(0.4)
