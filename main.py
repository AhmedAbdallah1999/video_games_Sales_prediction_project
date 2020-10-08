# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




df = pd.read_csv("vgsales.csv")
# using labelEncoder convert categorical data into numerical data
number = LabelEncoder()
df['Platform'] = number.fit_transform(df['Platform'].astype('str'))
df['Genre'] = number.fit_transform(df['Genre'].astype('str'))
df['Publisher'] = number.fit_transform(df['Publisher'].astype('str'))

dff = df.drop(['Rank', 'Name', 'Year'], axis=1)

df3 = dff.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)

# columns = ["Platform", "Genre", "Publisher"]
# columns = ["Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales"]
columns = ["Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales"]

labels = df3["Global_Sales"].values
features = dff[list(columns)].values

regr = linear_model.LinearRegression()

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

scaler = StandardScaler()
# scaler = preprocessing.MinMaxScaler()

# Fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)

regr.fit(X_train, y_train)

Accuracy = regr.score(X_train, y_train)
print("Accuracy in the training data: ", Accuracy * 100, "%")

accuracy = regr.score(X_test, y_test)
print("Accuracy in the test data", accuracy * 100, "%")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=200)

losses = pd.DataFrame(model.history.history)
losses.plot()

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score

predictions = model.predict(X_test)

print("deep learning :")
print(mean_absolute_error(y_test,predictions))
print(np.sqrt(mean_squared_error(y_test,predictions)))
print(explained_variance_score(y_test,predictions))
print(r2_score(y_test,predictions))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


model = Sequential()

model.add(Dense(X_train.shape[1],activation='relu'))
model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(0.001), loss='mse')


r = model.fit(X_train, y_train,
              validation_data=(X_test,y_test),
              batch_size=128,
              epochs=500)

losses = pd.DataFrame(model.history.history)
losses.plot()

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score

predictions = model.predict(X_test)

print("deep learning :")
print(mean_absolute_error(y_test,predictions))
print(np.sqrt(mean_squared_error(y_test,predictions)))
print(explained_variance_score(y_test,predictions))
print(r2_score(y_test,predictions))



