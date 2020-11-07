#First Machine Learning Model
#KNN  
#Determines the class of a car depending on it's buying power, maintenance and safety
#Data file is provided in repository

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Import and read data. In csv format
data = pd.read_csv('car.data')

#if there are any whitespaces in your data must include them as ' maintenance'
#not 'maintenance'
x = data[['buying', ' maintenance', ' safety']].values
y = data[['class']]

#converting the data using LabelEncoder
le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = le.fit_transform(x[:, i])

#converting y, assigning a value
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)


#create model
#Split the data into two parts, training and testing data
#Uses 80%-20% rule
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights= 'uniform')

#trains object, need x and y (features and labels)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction)
print("predictions:", prediction)
print("accuracy:", accuracy)


a = 539
print("actual value: ", y[a])
print("predicted value: ", knn.predict(x)[a])
