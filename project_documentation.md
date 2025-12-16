# EMG Signal Analysis and Muscle Activation Prediction Project

## Overview

This project focuses on EMG (Electromyography) signal analysis for real-time muscle activation prediction to control prosthetic arms. The system collects EMG signals from sensors, filters the raw data, extracts features, trains machine learning models to classify hand gestures (rest, moving right, left, up, down), and performs real-time prediction for prosthetic control.

The project includes:

- Data collection from Arduino sensors
- Signal filtering and preprocessing
- Feature extraction
- Machine learning model training (Random Forest, SVM, CNN, KNN, etc.)
- Real-time prediction with serial communication
- Arduino code for sensor reading and actuator control

## Project Structure

The workspace contains Python scripts, Arduino code, datasets (CSV/TXT), trained models (.joblib/.sav/.pkl/.h5), and other files.

## Code Files

### Data Collection Scripts

#### `data_collection.py`

Collects EMG data from Arduino via serial port and saves to text file.

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:27:20 2024

@author: jishn
"""

import serial

# Establish connection to the serial port that your Arduino is connected to.
# Replace '/dev/ttyUSB0' with the correct port for your Arduino.
ser = serial.Serial('COM3', 9600)

# Open text file to store the received data
with open("04BR.txt", 'w') as text_file:
    while True:
        # Read serial data from Arduino
        data = ser.readline().decode('utf-8').strip()
        if data:
            # Write the received data to the text file
            text_file.write(data + '\n')
            text_file.flush()
            print(f"Data written: {data}")
        else:
            break

# Close the serial connection
ser.close()
```

#### `faster data collection.py`

Similar to data_collection.py but optimized for faster collection.

#### `new data collection sayamandhhh.txt`

Text file containing data collection code (likely Python).

### Signal Filtering and Processing

#### `filteration_example.py`

Demonstrates EMG signal filtering, plotting, and basic gesture classification.

```python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:26:55 2024

@author: jishn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal_utilities as su
import emg as e

# Load EMG data
column_names = ['time', 'value']
emg_values = pd.read_csv('new data salman.csv', sep=',', names=column_names)

emg = emg_values['value']

time = emg_values['time'] / 1000

# Initialize the EMG filter
sample_frequency = 100 # Adjust as necessary
range_ = 0.1 # Adjust as necessary
min_EMG_frequency = 25 # Adjust as necessary
max_EMG_frequency = 600 # Adjust as necessary
reference_available = False # Adjust as necessary

# Instantiate the EMG_filter class
filter = e.EMG_filter(sample_frequency=sample_frequency, range_=range_, min_EMG_frequency=min_EMG_frequency, max_EMG_frequency=max_EMG_frequency, reference_available=reference_available)

# Apply the filter to each EMG value
filtered_emg_values = []
for value in emg:
    filtered_value = filter.filter(value)
    filtered_emg_values.append(filtered_value)
# Plot filtered EMG signal
plt.figure(figsize=(10, 6))
plt.plot(time, emg, label='Original EMG')
plt.plot(time, filtered_emg_values, label='Filtered EMG')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered EMG Signal')
plt.legend()
plt.grid(True)
plt.show()

# Convert EMG signal to DataFrame
filtered_emg_df = pd.DataFrame({'time': time, 'filtered_emg': filtered_emg_values})

mean_value = np.mean(filtered_emg_values)
std_value = np.std(filtered_emg_values)
print(mean_value)
print(std_value)

# Set the threshold based on the mean and standard deviation
threshold = std_value

# Classify the hand's state based on the threshold

filtered_emg_df['gesture'] = '0'
filtered_emg_df.loc[filtered_emg_df['filtered_emg'] >= threshold, 'gesture'] = '1'


# Save classified data to CSV
'''
output_file_path = '06B.csv'
filtered_emg_df.to_csv(output_file_path, index=False)
print("Filtered EMG data saved to:", output_file_path)
'''
```

#### `emg.py`

EMG filtering module with classes for basic and advanced filtering.

```python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:42:48 2024

@author: jishn
"""

import signal_utilities as su

#Simple filter class assumes no low-frequency noise and data centred at 0. More complex filteres inherit from this.
class EMG_filter_basic(object):
    def __init__(self, sample_frequency = 200, range_ = 0.1, reference_available = False):
        self.log = []                                          #used to store filtered data
        self.reference_available = reference_available
        self.MA = su.Moving_Average(length = sample_frequency * range_, return_int = True)

        if reference_available == True:
            self.LPF_data = su.LPF(cutoff_frequency = 150, sample_frequency = sample_frequency)
            self.LPF_reference = su.LPF(cutoff_frequency = 150, sample_frequency = sample_frequency)

    #input filtered data
    def log_data(self, to_log):
        self.log.insert(0, to_log)

    #pop and return either one data point or a set of data points
    def pop_log(self, num_to_pop = 1):
        if len(self.log) < 1 or num_to_pop > len(self.log):
            return
        elif num_to_pop <= 1:
            return self.log.pop()
        elif num_to_pop > 1 and num_to_pop <= len(self.log):
            to_return = self.log[-num_to_pop:]
            self.log = self.log[:-num_to_pop]
            return to_return

    #rectifies data, assuming low frequency noise already removed (reference point of 0)
    def rectify(self, to_rectify):
        return abs(to_rectify)

    #this function is called to input raw data. It returns a filtered value if it has enough samples, and also logs it
    def filter(self, data, reference_data = 0):
        if not self.reference_available:
            filtered_value = self.MA.get_movingAvg(self.rectify(data))
            self.log_data(filtered_value)
            return filtered_value

        else:
            clean_data = self.LPF_data.filter(data)
            clean_reference = self.LPF_reference.filter(reference_data)
            filtered_value = self.MA.get_movingAvg(self.rectify(clean_data - clean_reference))
            self.log_data(filtered_value)
            #print "Reference used"
            return filtered_value

#extension of EMG_filter_basic that has tools to remove low-frequency noise and normalize the data
class EMG_filter(EMG_filter_basic):

    def __init__(self, sample_frequency = 200, range_ = 0.5, min_EMG_frequency = 25, max_EMG_frequency = 150, reference_available = False):
        EMG_filter_basic.__init__(self, sample_frequency = sample_frequency, range_ = range_, reference_available = reference_available)
        #self.reference_available = reference_available
        self.PkPk = su.PkPk(sample_frequency = sample_frequency, min_frequency = min_EMG_frequency, max_frequency = max_EMG_frequency)

    #this function is called to input raw data and return a filtered value, accounting for low-frequency noise and un-normalized data
    def filter(self, data, reference_data = 0):
        if self.reference_available == True:
            clean_data = self.LPF_data.filter(data)
            clean_reference = self.LPF_reference.filter(reference_data)
            data = clean_data - clean_reference

        neutral_value = self.PkPk.get_pkpk(data)['neutral']
        filtered_value = self.MA.get_movingAvg(self.rectify(data - neutral_value))
        self.log_data(filtered_value)
        return filtered_value

        #return super(EMG_filter, self).filter(data - neutral_value)

#print "End of EMG module"
```

#### `signal_utilities.py`

Utility classes for signal processing: Moving Average, Low-Pass Filter, Peak-to-Peak detection.

```python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:42:19 2024

@author: jishn
"""

#Encapsulation of a key function that some filters require
class Moving_Average(object):
    def __init__(self, length, return_int = False):
        self.data = []
        self.data_sum = -1
        #self.data_avg = -1
        self.length = length
        self.value = -1
        self.return_int = return_int

        #self.sample_frequency = sample_frequency               #in Hz
        #self.range_ = range_                                   #in seconds
        #self.scope = 1.0 * self.sample_frequency * self.range_       #in number of samples, limits the length of movingAvg
        #self.sum_movingAvg = 0                                 #tracks the sum of the moving average
        #self.val_movingAvg = -1                                #the latest moving average value
        #self.movingAvg = []                                    #used to store the datapoints for taking a moving average

    def get_movingAvg (self, data):
        self.data.insert(0, data)
        self.data_sum += data

        if len(self.data) > self.length:
            self.data_sum -= self.data.pop()

        if self.return_int == True:
            self.value = int(self.data_sum / self.length) #preserves integer form
        else:
            self.value = 1.0 * self.data_sum / self.length

        if len(self.data) < (self.length / 2):
            return -1
        else:
            return self.value

#Simplifies the creation of a moving average-based LPF
class LPF(Moving_Average):
    def __init__(self, cutoff_frequency, sample_frequency, return_int = False):
        length = int(0.125 * sample_frequency / cutoff_frequency)
        Moving_Average.__init__(self, length, return_int = return_int)

    def filter(self, to_filter):
        if self.length < 2:
            return to_filter
        else:
            return self.get_movingAvg(to_filter)

class Basic_Stats(object):
    def __init__(self, length):
        self.length = length
        self.data_points = []
        self.total_sum = -1
        self.average = -1
        self.stddev = -1

    def add_data(self, data):
        self.data_points.insert(0, data)
        self.total_sum += data

        if len(self.data_points) > self.length:
            self.total_sum -= self.data_points.pop()

    def get_average(self, data):
        self.add_data(data)
        self.average = 1.0 * self.total_sum / len(self.data_points)
        return self.average


class PkPk(object):
    def __init__(self, sample_frequency, min_frequency, max_frequency):
        self.for_pkpk = []
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_pk_gap = sample_frequency / max_frequency
        self.max_pk_gap = sample_frequency / min_frequency

        #self.neutral = -1                                                   #signal is shifted so that point point is zero
        #self.min_EMG_frequency = min_EMG_frequency                          #signals below this frequency do not influence the peak to peak measurements
        #self.max_EMG_frequency = max_EMG_frequency                          #signals above this frequency do not influence the peak to peak measurements
        #self.min_pk_gap = 1.0/self.max_EMG_frequency * sample_frequency     #the minimun distance in data points that two registered peaks can be
        #self.max_pk_gap = 1.0/self.min_EMG_frequency * sample_frequency     #the maximum distance two consecutive peaks can be without the calculated neutral-point shifting significantly

        #self.pk_indices = []
        #self.pk_is_convex = []

    def get_pkpk(self, data):
        self.for_pkpk.insert(0, data)

        #discards any data beyond two periods of the lowest-frequency wave
        if len(self.for_pkpk) > (self.max_pk_gap * 2):
            self.for_pkpk.pop()

        highest = max(self.for_pkpk)
        lowest = min(self.for_pkpk)
        self.neutral = (highest + lowest)/2

        to_return = {'max' : highest, 'min' : lowest, 'pkpk' : highest - lowest, 'neutral' : self.neutral}

        if len(self.for_pkpk) < self.min_pk_gap * 2:
            return {'max' : -1, 'min' : -1, 'pkpk' : -1, 'neutral' : -1}
        else:
            return to_return


    def find_peaks(self):
        #cannot find another peak of oposite concavity until min_pk_gap data points past?
        pass

    def advanced_get_pkpk(self):
        pass

#print "End"
```

#### `just_filter.py`

Simple filtering script.

#### `plot.py`

Script for plotting EMG signals.

### Data Combination and Preparation

#### `combine_data.py`

Combines multiple CSV files into a single training dataset.

```python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:18:17 2024

@author: jishn
"""

import pandas as pd
import os

# List of CSV file names
csv_files = [
    "01Blf.csv",
    "01BRf.csv",
    "02Blf.csv",
    "02BRf.csv",
    "02Blf.csv",
    "02BRf.csv",
    "02Blf.csv",
    "02BRf.csv",
    "combined_data.csv"
]

# Initialize an empty list to hold the DataFrames
dfs = []

# Read each CSV file into a DataFrame and append it to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Drop the 'user' column if it exists
if 'user' in combined_df.columns:
    combined_df = combined_df.drop('user', axis=1)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("New Data for Training.csv", index=False)
```

#### `dataframe_for_model.py`

Prepares dataframes for model training by combining files and adding labels.

```python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:40:44 2024

@author: jishn
"""
'''
import pandas as pd

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Loop over the file names
for i in range(1, 6):
    # Read the data from the file
    file_name = f"filtered_emg_data0{i}.csv"
    data = pd.read_csv(file_name)

    # Add a new column for the user
    data['user'] = i

    # Reorder the columns
    data = data[['time', 'user', 'filtered_emg', 'gesture']]

    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data])

# Save the combined data to a new file
combined_data.to_csv("combined_data.csv", index=False)
'''

"""
import pandas as pd

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Loop over the file names
for i in range(1, 6):
    # Read the data from the file
    file_name = f"moredata0{i}.txt"
    data = pd.read_csv(file_name)



    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data])

# Save the combined data to a new file
combined_data.to_csv("combined_rawdata.csv", index=False)
"""


import pandas as pd

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Loop over the file names
for i in range(1, 8):
    # Read the data from the file
    file_name = f"handright0{i}.txt"
    data = pd.read_csv(file_name, sep=',', header=None, names=['time', 'value'])

    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data])

# Save the combined data to a new file
combined_data.to_csv("combined_raw_right_data.csv", index=False)
```

### Machine Learning Training Scripts

#### `cnn.py`

Contains CNN model training code (commented) and active Random Forest/SVM training with grid search.

```python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:26:32 2024

@author: jishn
"""
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Load your dataset
df = pd.read_csv("combined_all_data_new_with_more_columns.csv")

# Preprocess the data
X = df.drop(labels=["gesture"], axis=1).values
y = df["gesture"].values

# Convert labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Reshape X for 1D convolution
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # Assuming 5 classes for gestures

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
"""


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("combined_all_data_new_with_more_columns.csv")

# Handle missing values
df = df.dropna()

# Map categorical labels to numeric values
label_mapping = {
    "Hand is at Rest": 0,
    "Hand is Moving RIGHT": 1,
    "Hand is Moving LEFT": 2,
    "Hand is Moving UP": 3,
    "Hand is Moving DOWN": 4
}
df['gesture'] = df['gesture'].map(label_mapping)

# Define features (X) and target (Y)
X = df.drop(columns=["gesture"])
Y = df["gesture"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameters grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search cross-validation for Random Forest
print("Performing grid search for Random Forest...")
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)

# Best Random Forest model
best_rf_model = rf_grid_search.best_estimator_

# Evaluate Random Forest model on the test set
rf_predictions = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)

# Print best hyperparameters for Random Forest
print("Best Random Forest hyperparameters:", rf_grid_search.best_params_)
print()

# SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Define hyperparameters grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# Perform grid search cross-validation for SVM
print("Performing grid search for SVM...")
svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)

# Best SVM model
best_svm_model = svm_grid_search.best_estimator_

# Evaluate SVM model on the test set
svm_predictions = best_svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)

# Print best hyperparameters for SVM
print("Best SVM hyperparameters:", svm_grid_search.best_params_)
```

#### `knn.py`

KNN model training.

#### `Random_forest_model.py`

Random Forest training script.

#### `svm_model.py`

SVM training script.

### Real-time Prediction Scripts

#### `finalized_pred.py`

Real-time prediction using loaded SVM model, reads from serial, filters, predicts.

```python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:08:05 2024

@author: jishn
"""

from serial import Serial
import pandas as pd
import numpy as np
from joblib import load
import emg as e

# Load the trained SVM model
clf = load('finalized_model.sav') # Ensure this matches the saved model file name

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
            # Ensure the input is reshaped if necessary
            prediction = clf.predict([features])
            if prediction == 1:
                print("Hand is moving")
            else:
                print("Hand is at rest")
```

#### `realtime123.py`

Real-time prediction with Random Forest, sends prediction back to Arduino.

```python
# -- coding: utf-8 --
"""
Created on Wed Mar  5 12:09:53 2025

@author: user
"""

# -- coding: utf-8 --
"""
Updated on Wed Mar  5 2025

@author: Sayand
"""

import serial
import numpy as np
import pandas as pd
from joblib import load
import emg as e  # Assuming you have an 'emg' module with an EMG_filter class

# Load the trained Random Forest model
model = load("random_forest_new2.joblib")

# Set up serial communication with Arduino
ser = serial.Serial('COM6', 9600, timeout=1)  # Replace 'COM6' with the appropriate port

# Initialize the EMG filter
filter = e.EMG_filter(sample_frequency=100, range_=0.1, min_EMG_frequency=25, max_EMG_frequency=600, reference_available=False)

# Function to preprocess data and make prediction
def predict_hand_movement(time_stamp, emg_data):
    # Filter the EMG data
    filtered_emg = filter.filter(emg_data)
    # Create a DataFrame with the same column names as the training data
    data = pd.DataFrame([[time_stamp, filtered_emg]], columns=['time', 'filtered_emg'])
    # Predict hand movement
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
            print("Time:", time_stamp, "Raw EMG:", emg_value, "Filtered EMG:", filtered_emg, "Prediction:", movement_prediction)

            # Send the prediction (0 or 1) to the Arduino
            ser.write(f"{movement_prediction}\n".encode())

except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")
```

#### `random_prediciton.py`, `new_pred.py`, etc.

Various prediction scripts with different models and features.

### Other Scripts

- `hello.py`, `just.py`, `new.py`: Simple test scripts.
- `pred_for_knn.py`: KNN prediction.
- `synthetic_value_pred.py`: Synthetic data prediction.
- `Prosthetic control for 5 fingers.py`: Advanced prosthetic control.

## Arduino Code

#### `arduino.txt`

Basic Arduino code to read analog EMG sensor and send data over serial.

```plaintext
void setup() {
  Serial.begin(9600);
}

void loop() {
  unsigned long time = millis(); // get the current time in milliseconds
  int sensorValue = analogRead(A0);
  Serial.print(time);
  Serial.print(",");
  Serial.print(sensorValue);
  Serial.println();
  delay(100);
}

#save this in .ino format. This is an arduino code.
```

#### `arduino_code_with_buzzer.txt`

(Note: This appears to be Python code, not Arduino. It might be misnamed.)

```plaintext
from serial import Serial
import numpy as np
import pandas as pd
from joblib import load
import emg as e

# Load the trained Random Forest model
model = load("random_forest_new.joblib")

# Set up serial communication with Arduino
ser = Serial('COM5', 9600) # Replace 'COMX' with the appropriate port

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
            # Send signal to Arduino based on filtered EMG value
            if movement_prediction == 0:
                print("hand is at rest")
            else:
                print("hand is moving")
            print(filtered_emg)
            if movement_prediction == 1:
                if filtered_emg > 95:
                    ser.write(b'6') # Loud beep
                else:
                    ser.write(b'1') # Normal beep
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")
```

## Datasets

The project contains numerous CSV and TXT files with EMG data:

### Raw Data Files (TXT)

- `handdown01.txt` to `handdown07.txt`: Raw EMG data for hand moving down gesture.
- `handleft01.txt` to `handleft07.txt`: Raw EMG data for hand moving left.
- `handrest01.txt` to `handrest07.txt`: Raw EMG data for hand at rest.
- `handright01.txt` to `handright07.txt`: Raw EMG data for hand moving right.
- `handup01.txt` to `handup07.txt`: Raw EMG data for hand moving up.
- `01Bl.txt`, `01BR.txt`, `02Bl.txt`, etc.: Additional raw data files.
- `diffdata01.txt` to `diffdata03.txt`: Differential data.
- `moredata01.txt` to `moredata06.txt`: More raw data.
- `newdata01.txt` to `newdata03.txt`: New raw data.
- `updated*.txt`: Updated data files.

### Filtered/Processed Data Files (CSV)

- `filtered_emg_data01.csv` to `filtered_emg_data05.csv`: Filtered EMG data.
- `combined_*.csv`: Combined datasets for different gestures and users.
- `For paper.csv`, `For_paper_with_label.csv`: Data prepared for research paper.
- `New Data for Training.csv`: Combined training data.

### Combined Datasets

- `combined_all_data_new_with_more_columns.csv`: Main training dataset with features.
  Columns: time, filtered_emg, gesture, mean, std, min, max, median, zcr, ssc, wl, se
  Gestures: "Hand is at Rest", "Hand is Moving RIGHT", "Hand is Moving LEFT", "Hand is Moving UP", "Hand is Moving DOWN"

## Trained Models

The workspace contains many saved machine learning models:

### Joblib Models (.joblib)

- `random_forest.joblib`, `random_forest_new.joblib`, `random_forest_new2.joblib`, etc.: Random Forest models.
- `knn_model_with_window_5.joblib`, `sknn.joblib`, etc.: KNN models.
- `svm_model_with_window_5.joblib`, `new_svm_model.joblib`, etc.: SVM models.
- `latest_model.joblib`, `finalized_model.sav`: Final/best models.

### Other Formats

- `model.h5`: Keras/TensorFlow model (CNN).
- `lda_gesture_model.pkl`, `naive_bayes_model.pkl`: Other models.

## Usage Instructions

1. **Data Collection**: Run `data_collection.py` or `faster data collection.py` with Arduino connected to collect raw EMG data.

2. **Signal Processing**: Use `filteration_example.py` or `emg.py` to filter raw signals.

3. **Dataset Preparation**: Run `combine_data.py` and `dataframe_for_model.py` to prepare training datasets.

4. **Model Training**: Execute training scripts like `cnn.py`, `Random_forest_model.py`, etc., to train models on prepared datasets.

5. **Real-time Prediction**: Use scripts like `finalized_pred.py` or `realtime123.py` for real-time gesture prediction with Arduino.

6. **Arduino Setup**: Upload `arduino.txt` code to Arduino for sensor reading.

## Dependencies

- Python libraries: pandas, numpy, matplotlib, scikit-learn, joblib, serial, tensorflow/keras
- Hardware: Arduino board, EMG sensors

## Notes

- Many scripts have hardcoded file paths and parameters that may need adjustment.
- Serial ports (COM3, COM5, etc.) need to be configured based on system setup.
- Models are trained on specific datasets; retraining may be needed for new data.
- The project evolved over time with multiple contributors, hence various similar scripts.

This documentation covers the core components. For detailed implementation, refer to the individual code files.</content>
<parameter name="filePath">c:\Users\jishn\Documents\GitHub\emg_singal_analysis\project_documentation.md
