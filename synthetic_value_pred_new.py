# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:48:09 2025

@author: jishn
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:39:28 2025

@author: jishn
"""

import time
import numpy as np
import pandas as pd
import random
from collections import deque
from joblib import load
import emg as e  # Assuming you have the EMG filter class
import warnings
warnings.filterwarnings("ignore")


# Load the trained ML model and scaler
model = load("knn_smote_min.joblib")
scaler = load("scaler.joblib")

# Initialize EMG filter
sample_frequency = 100  # Adjust as necessary
range_ = 0.1
min_EMG_frequency = 25
max_EMG_frequency = 600
reference_available = False
emg_filter = e.EMG_filter(sample_frequency, range_, min_EMG_frequency, max_EMG_frequency, reference_available)

# Window setup for feature extraction
window_size = 5  # Window size for feature extraction
window = deque(maxlen=window_size)

# Function to generate values
def generate_values():
    min_value = 150
    max_value = 1024
    repeat_count = 10
    random_values_count = 25
    delay = 0.2

    sequence = []

    # Start at 150 and repeat 10 times
    sequence.extend([min_value] * repeat_count)

    # Generate 25 random values increasing from 150 to 1024
    current_value = min_value
    for _ in range(random_values_count):
        current_value = random.randint(current_value, max_value)
        sequence.append(current_value)

    # Repeat the last generated value 10 times
    sequence.extend([current_value] * repeat_count)

    # Decrease from the last value back to 150 in 20 steps
    step_size = (current_value - min_value) / 20
    for _ in range(20):
        current_value = int(current_value - step_size)
        sequence.append(current_value)

    # Stay at 150 for 10 repeats
    sequence.extend([min_value] * repeat_count)
    
    return sequence

# Run the sequence continuously
while True:
    sequence = generate_values()
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
