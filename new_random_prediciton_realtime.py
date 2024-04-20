# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:26:18 2024

@author: jishn
"""




import serial
import numpy as np
import pandas as pd
from joblib import load
import emg as e
import cv2
import threading

# Load the trained Random Forest model
model = load("random_forest_new2.joblib")

# Set up serial communication with Arduino
ser = serial.Serial('COM5', 9600) # Replace 'COMX' with the appropriate port

# Initialize the EMG filter
filter = e.EMG_filter(sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600,
                      reference_available=False)

# Global variable to control video playback loop
video_playing = False

# Function to preprocess data and make prediction
def predict_hand_movement(time_stamp, emg_data):
    # Filter the EMG data
    filtered_emg = filter.filter(emg_data)
    # Create a DataFrame with the same column names as the training data
    data = pd.DataFrame([[time_stamp, filtered_emg]], columns=['time', 'filtered_emg'])
    # Predict hand movement without scaling
    prediction = model.predict(data)
    return time_stamp, filtered_emg, prediction[0]

# Function to play video in a separate thread
def play_video():
    global cap, video_playing
    cap = cv2.VideoCapture("video1.mp4")
    while video_playing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Player', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            video_playing = False
    cap.release()
    cv2.destroyAllWindows()

# Function to open and close video
def open_close_video(is_open):
    global video_playing
    if is_open:
        video_playing = True
        threading.Thread(target=play_video).start()
    else:
        video_playing = False

# Main loop for real-time prediction
try:
    consecutive_count = 0
    video_open_count = 0
    is_video_open = False
    while True:
        # Read data from Arduino
        raw_data = ser.readline().decode('utf-8').rstrip()
        # Ensure raw_data is not empty
        if raw_data:
            # Split the data into time and EMG value
            time_stamp, emg_value = map(float, raw_data.split(','))
            # Make prediction
            time_prediction, filtered_emg, movement_prediction = predict_hand_movement(time_stamp, emg_value)
            if movement_prediction == 1:
                print("Hand is moving")
                if not is_video_open:
                    consecutive_count += 1
                    if consecutive_count == 10:
                        open_close_video(True)
                        is_video_open = True
                        consecutive_count = 0
                else:
                    video_open_count += 1
                    if video_open_count == 10:
                        open_close_video(False)
                        is_video_open = False
                        video_open_count = 0
            else:
                consecutive_count = 0
                video_open_count = 0
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")


    