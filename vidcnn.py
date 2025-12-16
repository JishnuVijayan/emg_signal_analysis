# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:23:32 2024

@author: jishn
"""
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Function to load and preprocess a single dataset
def load_and_preprocess_data(file_name):
    df = pd.read_csv(file_name)
    scaler = MinMaxScaler()
    df['emg'] = scaler.fit_transform(df[['emg']])
    encoder = LabelEncoder()
    df['gesture'] = encoder.fit_transform(df['gesture'])
    return df

# List of datasets
datasets = [
    'combined_emg_hand_right2_data_vid.csv',
    'combined_emg_hand_left_data_vid.csv',
    'combined_emg_hand_up_data_vid.csv',
    'combined_emg_hand_down_data_vid.csv'
]

# Load and preprocess each dataset
dataframes = [load_and_preprocess_data(file_name) for file_name in datasets]

# Combine the datasets
combined_df = pd.concat(dataframes, ignore_index=True)

# Split the combined dataset
X = combined_df.drop(['gesture'], axis=1)
y = combined_df['gesture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Design the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')  # 4 gestures: right, left, up, down
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Calculate and print the average accuracy over the epochs
average_accuracy = np.mean(history.history['accuracy'])
print(f"Average Accuracy: {average_accuracy:.2f}%")

# Save the model
model.save('model.h5')
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to load and preprocess a single dataset
def load_and_preprocess_data(file_name):
    df = pd.read_csv(file_name)
    print(f"Columns in {file_name}: {df.columns}")  # Print columns to diagnose issues
    if 'filtered_emg' not in df.columns:
        raise ValueError(f"'filtered_emg' column not found in {file_name}. Please check the file.")
    scaler = MinMaxScaler()
    df['filtered_emg'] = scaler.fit_transform(df[['filtered_emg']])
    encoder = LabelEncoder()
    df['gesture'] = encoder.fit_transform(df['gesture'])
    return df

# List of datasets
datasets = [
    'combined_emg_hand_right2_data_vid.csv',
    'combined_emg_hand_left_data_vid.csv',
    'combined_emg_hand_up_data_vid.csv',
    'combined_emg_hand_down_data_vid.csv'
]

# Load and preprocess each dataset
dataframes = [load_and_preprocess_data(file_name) for file_name in datasets]

# Combine the datasets
combined_df = pd.concat(dataframes, ignore_index=True)

# Split the combined dataset
X = combined_df.drop(['gesture', 'time'], axis=1)  # Assuming 'time' is not needed for the model
y = combined_df['gesture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Design a simpler model without Conv1D layers
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # 4 gestures: right, left, up, down
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Calculate and print the average accuracy over the epochs
average_accuracy = np.mean(history.history['accuracy'])
print(f"Average Accuracy: {average_accuracy:.2f}%")

# Save the model
model.save('model.h5')
