# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:22:15 2022

@author: S Suhaasini
"""
#importing the libraries
import os
os.getcwd()
os.chdir('D:\Badhra\BE\Exposys Project\MY Project\CODE')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset using the Pandas library
dataset= pd.read_csv('50_Startups.csv')

#extracting matrix features
X= dataset.iloc [:,:-1]. values
Y= dataset.iloc [: , 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Backward elimination
import statsmodels.api as sm

# append column to independent variable X matrix
X=np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)

X_opt=X[:, [0,1,2,3]]
regressor_OLS=sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()

# remove x2 since P > 0.5
X_opt=X[:, [0,1,3]]
regressor_OLS=sm.OLS(endog = Y, exog=X_opt).fit()
regressor_OLS.summary()
