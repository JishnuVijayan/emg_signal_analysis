# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:05:32 2024

@author: jishn
"""

import pandas as pd
import numpy as np

combined_df = pd.read_csv('New_Data.csv')
#print(combined_df)

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar.fit(combined_df.drop('gesture', axis=1))
scaled_features = scalar.transform(combined_df.drop('gesture',axis=1))


df_features = pd.DataFrame(scaled_features, columns=combined_df.columns[:-1])
print(df_features.head())

from sklearn.model_selection import train_test_split

X = df_features
y = combined_df['gesture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

#error_rate = []
#for i in range(1, 40):
 #   knn = KNeighborsClassifier(n_neighbors=i)
 #   knn.fit(X_train, y_train)
 #   pred_i = knn.predict(X_test)
 #   error_rate.append(np.mean(pred_i != y_test))
    
#plt.figure(figsize=(10, 6))
#plt.plot(range(1, 40), error_rate, color='blue', marker='o')
#plt.show()


#from joblib import dump

#feature_imp = pd.Series(knn.feature_importances_, index=feature_names).sort_values(ascending=False)
#print(feature_imp)

#filename = "KNN_MODEL.joblib"
#dump((knn, feature_names), filename)

#import shap

# Create a SHAP explainer for the kNN model
#explainer = shap.Explainer(knn.predict, X_test)
#shap_values = explainer(X_test)

# Plot summary of SHAP values
#shap.summary_plot(shap_values, X_test)

from joblib import dump

filename = "KNN_MODEL.joblib"
dump(knn, filename)
print(f"Model saved as {filename}")


from joblib import dump
dump((knn, scalar), "KNN_MODEL_WITH_SCALER.joblib")

