import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report

dataset = pd.read_csv("bill_authentication.csv")

target  = dataset.pop("Class")

x = dataset

y = target

#splitting data for traing and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# #defing the model
svmClassfier = SVC(kernel='linear')

# #training the model
svmClassfier.fit(X_train, y_train)
# print(svmClassfier.score(X_train, y_train))

# #testing the model
y_pred = svmClassfier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

