# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:08:55 2024

@author: jishn
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("combined_filtered_data.csv")

#print(df.head())

#Calculating the count of 'hand is at rest' and 'hand is moving'
"""
sizes = df['gesture'].value_counts(sort = 1)
print(sizes)
"""

#Removing rows is value is null
df = df.dropna()

df.loc[df.gesture == "Hand is at Rest", 'gesture'] = 0
df.loc[df.gesture == "Hand is Moving", 'gesture'] = 1
print(df.head())


#Setting Y value which is dependent value that we are going to predict
Y = df["gesture"].values

#since the datatype of Y is object we are converting it into int32
Y = Y.astype("int")

#X is independent value
X = df.drop(labels = ["gesture"], axis = 1)
#print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state=42)
#random_state can be any integer and it is used as a seed to randomly split dataset.
#By doing this we work with same test dataset evry time, if this is important.
#random_state=None splits dataset randomly every time


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 10, random_state = 20)

model.fit(X_train, y_train)

#Calculating the accuracy

prediction_test = model.predict(X_test)


from sklearn import metrics
#comparing prediction of y_test and prediction_test for calculating accuray
print("Accuracy = ",metrics.accuracy_score(y_test, prediction_test))


feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending=False)
print(feature_imp)

from joblib import dump

filename = "random_forest_new.joblib"

dump(model, filename)

from sklearn.preprocessing import StandardScaler


# Assuming X_train is your training data
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler
dump(scaler, 'scaler_new.joblib')

