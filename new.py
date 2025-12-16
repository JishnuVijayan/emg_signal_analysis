# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:21:17 2024

@author: jishn
"""

import pandas as pd
import numpy as np
import os

# Function to create windows of data
def create_windows(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i+window_size]
        windows.append(window)
    return windows

# Define window parameters
window_size = 30  # Adjust as needed
step_size = 15    # Adjust as needed

# Read filtered EMG data
df = pd.read_csv('combined_all_data.csv')

# Create windowed data
windowed_data = []
for gesture in df['gesture'].unique():
    gesture_data = df[df['gesture'] == gesture]['filtered_emg']
    windows = create_windows(gesture_data, window_size, step_size)
    for window in windows:
        windowed_data.append({'window': window, 'gesture': gesture})

# Extract features from windowed data
features = []
for data in windowed_data:
    window = data['window']
    mean = np.mean(window)
    std_dev = np.std(window)
    max_val = np.max(window)
    min_val = np.min(window)
    features.append({'mean': mean, 'std_dev': std_dev, 'max': max_val, 'min': min_val, 'gesture': data['gesture']})

# Create DataFrame from features
features_df = pd.DataFrame(features)

# Train machine learning model (Random Forest classifier example)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = features_df.drop(columns=['gesture'])
y = features_df['gesture']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from joblib import dump

filename = "n.joblib"

dump(model, filename)
