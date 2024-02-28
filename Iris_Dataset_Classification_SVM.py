import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report

iris = pd.read_csv("Iris.csv")

x = iris.iloc[:,:-1]
y = iris.iloc[:, -1].values

#splitting data for traing and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# #defing the model
svmClassfier = SVC(kernel='poly',degree=3)
# svmClassfier =SVC(kernel='rbf', C=1, gamma=1)

#training the model
svmClassfier.fit(X_train, y_train)

# #testing the model
y_pred = svmClassfier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

