
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = pd.read_csv("housing.csv")

housing.pop("ocean_proximity")
housing.pop("total_bedrooms")

hnorm = (housing - housing.min() - 1)/ (housing.max() - housing.min())

target_column = "median_house_value"
target = housing.pop(target_column)

y = target.values
x = hnorm.values

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3)

model = keras.models.Sequential([keras.layers.Dense(16, activation="relu", 
                                input_shape=(8,)), keras.layers.Dense(1)])

model.compile(loss="mae" , optimizer = "sgd")

history = model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test))

plt.plot(history.history['loss'] , label = 'Training Loss')
plt.plot(history.history['val_loss'] , label = 'Validation Loss')
plt.title('Training and Validation Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

evalution_result=model.evaluate(x_test,y_test)

print("Test loss: ",evalution_result)
