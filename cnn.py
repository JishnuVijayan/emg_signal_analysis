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
