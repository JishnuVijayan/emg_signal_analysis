# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:08:55 2024

@author: jishn

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump

# Define a function to create a rolling window
def create_rolling_window(df, window_size):
    feature_df = df.drop(labels=["gesture"], axis=1)
    temp_df = feature_df.copy()
    for i in range(window_size):
        feature_df = pd.concat([feature_df, temp_df.shift(-(i + 1))], axis=1)

    feature_df.dropna(axis=0, inplace=True)
    return feature_df

# Load the data
df = pd.read_csv("combined_all_data_new_with_more_columns.csv")

# Print the count of each gesture
sizes = df['gesture'].value_counts(sort=1)
print(sizes)

# Remove rows with null values
df = df.dropna()

# Map gestures to integers
df.loc[df.gesture == "Hand is at Rest", 'gesture'] = 0
df.loc[df.gesture == "Hand is Moving RIGHT", 'gesture'] = 1
df.loc[df.gesture == "Hand is Moving LEFT", 'gesture'] = 2
df.loc[df.gesture == "Hand is Moving UP", 'gesture'] = 3
df.loc[df.gesture == "Hand is Moving DOWN", 'gesture'] = 4

# Define the window size
window_size = 10

# Apply the function to your dataframe
X = create_rolling_window(df, window_size)

# Store the column names for future reference
feature_names = X.columns.tolist()

# Truncate the 'gesture' column to match the length of X
Y = df["gesture"].values[:len(X)]
Y = Y.astype("int")  # Convert Y to int

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=20, max_depth=50, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, bootstrap=False)
model.fit(X_train, y_train)

# Make predictions on the test set
prediction_test = model.predict(X_test)

# Print the accuracy
print("Accuracy =", metrics.accuracy_score(y_test, prediction_test))

# Print the feature importances
feature_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print(feature_imp)

# Save the trained model and feature names
filename = "latest_model_nnnn.joblib"
dump((model, feature_names), filename)





"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Define a function to create a rolling window
def create_rolling_window(df, window_size):
    feature_df = df.drop(labels = ["gesture"], axis = 1)
    temp_df = feature_df.copy()
    for i in range(window_size):
        feature_df = pd.concat([feature_df, temp_df.shift(-(i+1))], axis = 1)
        
    feature_df.dropna(axis=0, inplace=True)
    return feature_df

# Load the data
df = pd.read_csv("combined_all_data_new_with_more_columns.csv")

# Print the count of each gesture
sizes = df['gesture'].value_counts(sort = 1)
print(sizes)

# Remove rows with null values
df = df.dropna()

# Map gestures to integers
df.loc[df.gesture == "Hand is at Rest", 'gesture'] = 0
df.loc[df.gesture == "Hand is Moving RIGHT", 'gesture'] = 1
df.loc[df.gesture == "Hand is Moving LEFT", 'gesture'] = 2
df.loc[df.gesture == "Hand is Moving UP", 'gesture'] = 3
df.loc[df.gesture == "Hand is Moving DOWN", 'gesture'] = 4

# Define the window size
window_size = 10

# Apply the function to your dataframe
X = create_rolling_window(df, window_size)

# Truncate the 'gesture' column to match the length of X
Y = df["gesture"].values[:len(X)]
Y = Y.astype("int")  # Convert Y to int

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['sqrt', None],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


# Create a base model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Train the model using the best parameters
best_grid = grid_search.best_estimator_
best_grid.fit(X_train, y_train)

# Make predictions on the test set
prediction_test = best_grid.predict(X_test)

# Print the accuracy
print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

# Print the feature importances
feature_list = list(X.columns)
feature_imp = pd.Series(best_grid.feature_importances_, index = feature_list).sort_values(ascending=False)
print(feature_imp)
"""

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump
# Define a function to create a rolling window
def create_rolling_window(df, window_size):
    feature_df = df.drop(labels=["gesture"], axis=1)
    temp_df = feature_df.copy()
    for i in range(window_size):
        feature_df = pd.concat([feature_df, temp_df.shift(-(i + 1))], axis=1)

    feature_df.dropna(axis=0, inplace=True)
    return feature_df

# Load the data
df = pd.read_csv("combined_right_and_up_data.csv")

# Print the count of each gesture
sizes = df['gesture'].value_counts(sort=1)
print(sizes)

# Remove rows with null values
df = df.dropna()

# Map gestures to integers
df.loc[df.gesture == "Hand is at Rest", 'gesture'] = 0
df.loc[df.gesture == "Hand is Moving RIGHT", 'gesture'] = 1
df.loc[df.gesture == "Hand is Moving UP", 'gesture'] = 2


# Define the window size
window_size = 10

# Apply the function to your dataframe
X = create_rolling_window(df, window_size)

# Truncate the 'gesture' column to match the length of X
Y = df["gesture"].values[:len(X)]
Y = Y.astype("int")  # Convert Y to int

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=20, max_depth=50, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, bootstrap=False)
model.fit(X_train, y_train)

# Make predictions on the test set
prediction_test = model.predict(X_test)

# Print the accuracy
print("Accuracy =", metrics.accuracy_score(y_test, prediction_test))

# Print the feature importances
feature_imp = pd.Series(model.feature_importances_, index=X.columns.tolist()).sort_values(ascending=False)
print(feature_imp)

# Save the trained model and feature names
filename = "latest_model_with_right_and_up.joblib"
dump((model, X.columns.tolist()), filename)
"""