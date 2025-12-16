

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:39:53 2024

@author: jishn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from joblib import dump
import emg as e

# Load and process EMG data from multiple files
file_paths = ['moredata01.txt', 'moredata02.txt', 'moredata03.txt','moredata05.txt']
dfs = []

for file_path in file_paths:
    column_names = ['time', 'value']
    emg_values = pd.read_csv(file_path, sep=',', names=column_names)
    emg = emg_values['value']
    time = emg_values['time'] / 1000

    # Initialize the EMG filter
    filter = e.EMG_filter(sample_frequency=1000, range_=0.5, min_EMG_frequency=25, max_EMG_frequency=450, reference_available=False)

    # Apply the filter to each EMG value
    filtered_emg_values = []
    for value in emg:
        filtered_value = filter.filter(value)
        filtered_emg_values.append(filtered_value)

    # Convert EMG signal to DataFrame
    filtered_emg_df = pd.DataFrame({'time': time, 'filtered_emg': filtered_emg_values})
    dfs.append(filtered_emg_df)

# Concatenate data from all datasets
combined_df = pd.concat(dfs, ignore_index=True)

# Define features (e.g., mean)
def calculate_features(emg_values):
    mean = np.mean(emg_values)
    return [mean]

# Calculate features for each window
window_size = 5 # Updated window size to 10
feature_cols = ['mean'] # Updated feature column to 'mean'
features = []
for i in range(0, len(combined_df), window_size):
    window = combined_df['filtered_emg'].iloc[i:i+window_size]
    if len(window) == window_size:
        features.append(calculate_features(window))

# Create DataFrame for features
features_df = pd.DataFrame(features, columns=feature_cols)

# Split data into features and target
X = features_df.values
y = np.where(features_df['mean'] >= features_df['mean'].mean(), 1, 0) # Example thresholding

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for the SVM, including probability=True
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly'], 'gamma': [0.1, 1, 10], 'probability': [True]}

# Initialize the SVC with probability=True
clf = SVC(probability=True, random_state=42)

# Perform grid search to find the best parameters
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Train the SVM with the best parameters
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = best_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict probabilities for test set
yhat_probs = best_clf.predict_proba(X_test)[:, 1]

# Predict crisp classes for test set
yhat_classes = best_clf.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, yhat_classes)
precision = precision_score(y_test, yhat_classes)
recall = recall_score(y_test, yhat_classes)
f1 = f1_score(y_test, yhat_classes)
kappa = cohen_kappa_score(y_test, yhat_classes)
auc = roc_auc_score(y_test, yhat_probs)
matrix = confusion_matrix(y_test, yhat_classes)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1}')
print(f'Cohen\'s kappa: {kappa}')
print(f'ROC AUC: {auc}')
print(f'Confusion Matrix:\n{matrix}')

# Save the trained model
dump(best_clf, 'svm_model.joblib')
