#!/usr/bin/env python
# coding: utf-8

# # **Importing the dependencies**

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# # **Data Collection And Analysis**

# ## PIMA Diabetes Dataset

# In[ ]:


# loading the diabetes dataset to pandas dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv' , )

# In[ ]:


 #pd.read_csv?

# In[ ]:


#Printing the first five 5 rows of the dataset
diabetes_dataset.head()

# In[ ]:


# Rows and Columns in this Dataset
diabetes_dataset.shape

# In[ ]:


# getting the statistical measure of the data
diabetes_dataset.describe()

# In[ ]:


diabetes_dataset['Outcome'].value_counts()

# # **0 ---> NON DIABETIC**
# # **1--->>> DIABETIC**
# 

# In[ ]:


diabetes_dataset.groupby('Outcome').mean()

# In[ ]:


# seprating data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# In[ ]:


print(X)

# In[ ]:


print(Y)

# # **Data Standardrization**

# In[ ]:


Scaler = StandardScaler()

# In[ ]:


Scaler.fit(X)

# In[ ]:


Standardrized_data = Scaler.transform(X)

# In[ ]:


print(Standardrized_data)

# In[ ]:


X = Standardrized_data
Y = diabetes_dataset['Outcome']

# In[ ]:


print(X)
print(Y)

# # **Train Test Split**

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# In[ ]:


print(X.shape, X_train.shape, X_test.shape)

# # **Training model**

# In[ ]:


Classifier = svm.SVC(kernel='linear')

# In[ ]:


#training the svm classifier
Classifier.fit(X_train, Y_train)

# # **Model Evaluation**

# # **Accuracy Score**

# In[ ]:


# Accuracy score on the training data
X_train_prediction = Classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# In[ ]:


print("Accuracy Score of the training data :", training_data_accuracy)

# In[ ]:


# Accuracy score on the test data
X_test_prediction = Classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# In[ ]:


print("Accuracy Score of the test data :", test_data_accuracy)

# # **Making a predictive system**

# In[ ]:


#input_data = (10,115,0,0,0,35.3,0.134,29)
#input_data = (1,89,66,23,94,28.1,0.167,21)
input_data = (11, 143, 94, 33, 146, 36.6, 0.254, 51)
# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardrized the input data
std_data = Scaler.transform(input_data_reshaped)
print(std_data)

prediction = Classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print("The Person is not Diabetic")
else:
    print("The Person is  Diabetic")

