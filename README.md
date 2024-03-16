# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/Visalan-H/nn-classification/assets/152077751/30b594b8-992a-41d7-b2d1-f983d4c29515)


## DESIGN STEPS

1)Load and Preprocess Data: Load the dataset "customers.csv" and handle missing values by dropping rows with null values. Encode string data types into numerical values.

2)Exploratory Data Analysis (EDA): Analyze the dataset using correlation matrix and heatmap. Visualize data relationships using various plots like pairplot, displot, countplot, and scatterplot.

3)Data Splitting: Split the dataset into training and testing sets using train_test_split.

4)Model Creation: Design a neural network model with 2 hidden layers of 6 and 16 neurons each and an output layer with 4 neurons for multi-classification.

5)Model Training: Compile and fit the model with the training data.

6)Model Validation: Validate the model's performance using training data, by calculating the training accuracy.

7)Model Evaluation: Evaluate the model's performance using a confusion matrix on the testing data.


## PROGRAM

### Name: Visalan H
### Register Number:212223240183

```python
<a href="https://colab.research.google.com/github/Rithigasri/DL_Exp02/blob/main/DL_EXP02.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt


customer_df = pd.read_csv('customers.csv')

customer_df.columns

customer_df.dtypes

customer_df.shape

customer_df.isnull().sum()

customer_df_cleaned = customer_df.dropna(axis=0)

customer_df_cleaned.isnull().sum()

customer_df_cleaned.shape

customer_df_cleaned.dtypes

customer_df_cleaned['Gender'].unique()

customer_df_cleaned['Ever_Married'].unique()

customer_df_cleaned['Graduated'].unique()

customer_df_cleaned['Profession'].unique()

customer_df_cleaned['Spending_Score'].unique()

customer_df_cleaned['Var_1'].unique()

customer_df_cleaned['Segmentation'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)


customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])

customers_1.dtypes

le = LabelEncoder()

customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)

customers_1.dtypes

# Calculate the correlation matrix
corr = customers_1.corr()

# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

sns.pairplot(customers_1)

sns.distplot(customers_1['Age'])

plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])

plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=customers_1)

customers_1.describe()

customers_1['Segmentation'].unique()

X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values



y1 = customers_1[['Segmentation']].values

one_hot_enc = OneHotEncoder()

one_hot_enc.fit(y1)

y1.shape

y = one_hot_enc.transform(y1).toarray()

y.shape

y1[0]

y[0]

X.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)

X_train[0]

X_train.shape

scaler_age=MinMaxScaler()

scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

model=Sequential([
    Dense(units=6,activation='relu',input_shape=[8]),
    Dense(units=16,activation='relu'),
    Dense(units=4,activation='softmax')
])

model.compile(optimizer='adam',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])

model.fit(x=X_train_scaled,y=y_train,
             epochs=1000,
             batch_size= 256,
             validation_data=(X_test_scaled,y_test),
             )

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

x_test_predictions.shape

y_test_truevalue = np.argmax(y_test,axis=1)

y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

# Saving the Model
model.save('customer_classification_model.h5')


# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

# Loading the Model
model = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

x_single_prediction = np.argmax(model.predict(X_test_scaled[1:2,:]), axis=1)

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))



## Dataset Information

Include screenshot of the dataset

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
Include your plot here

### Classification Report

Include Classification Report here

### Confusion Matrix

Include confusion matrix here


### New Sample Data Prediction

Include your sample input and output here

## RESULT
Include your result here
