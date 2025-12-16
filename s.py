# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:58:08 2024

@author: jishn
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import numpy as np

# Load your filtered EMG data
data = pd.read_csv('filtered_emg_data.csv')

# Preprocessing and feature engineering steps remain the same
X = data[['filtered_emg']]
y = data['gesture']
y = y.map({'Hand is Moving': 1, 'Hand is at Rest': 0})

# Define the parameter grid for the C values you want to test
param_grid = {
    'svc__C': [0.1, 1, 10, 100, 1000],
    'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'svc__kernel': ['rbf']
}

# Set up the pipeline with SMOTE for handling class imbalance and GridSearchCV for hyperparameter tuning
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('svc', SVC())
])

grid = GridSearchCV(pipeline, param_grid, refit=True, verbose=3)

# Fit the model to your training data
grid.fit(X, y)

# After fitting, you can inspect the best parameters found by GridSearchCV
print("Best parameters found:", grid.best_params_)

# And use the best estimator for predictions
best_svm_model = grid.best_estimator_
y_pred_svm = best_svm_model.predict(X)

# Finally, evaluate your model
print("SVM Confusion Matrix:", confusion_matrix(y, y_pred_svm))
print("SVM Classification Report:", classification_report(y, y_pred_svm))
print("SVM AUC-ROC:", roc_auc_score(y, y_pred_svm))

# Save the model using joblib
filename = 'finalized_model.sav'
dump(best_svm_model, filename)
print(f"Model saved to: {filename}")
