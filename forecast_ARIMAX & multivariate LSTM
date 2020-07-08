#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 00:59:16 2020
@author: alexchengjr
"""

#%%
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow
import keras

#%%
us_data = pd.read_csv('correlation_us.csv')
us_data.head()

#%%
us_data['correlation_us'].plot()
plt.show()

us_data.loc[3200:,['new_us']].plot()
plt.show()

us_data['vol_stock_us'].plot()
plt.show()

us_data['vol_commo_us'].plot()
plt.show()

us_data['vix'].plot()
plt.show()

#%% ADF Test
from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    print('===========================================')
    print('Result of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value',\
                         '#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    print('===========================================')
    
adf_test(us_data['correlation_us'])

#%% KPSS Test
from statsmodels.tsa.stattools import kpss

def kpss_test(timeseries):
    print('==================================')
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3],\
                            index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    print('==================================')

kpss_test(us_data['correlation_us'])

#%% Variables
covidf_us = pd.read_csv('forecast_us.csv')
covidf_us.head()
covidf_us = covidf_us.loc[:,['forecast_case','vol_stock_us_fcst',\
                             'vol_commo_us_fcst','forecast_vix']]
covidf_us.columns = ['new_us','vol_stock_us','vol_commo_us','vix']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

endog_us = us_data.loc[:,['correlation_us']]
exog_us = us_data.loc[:,['new_us','vol_stock_us','vol_commo_us','vix']]
exog_us = pd.concat([exog_us, covidf_us], ignore_index=True, sort=False)
exog_us = scaler.fit_transform(exog_us)
exog_us = pd.DataFrame(data=exog_us[0:,0:],\
                       columns=['new_us','vol_stock_us','vol_commo_us','vix'])

endog_train = endog_us.loc[:3389,'correlation_us']
exog_train = exog_us.loc[:3389,['new_us','vol_stock_us','vol_commo_us','vix']]

endog_test = endog_us.loc[3277:3389,'correlation_us']
exog_test = exog_us.loc[3277:3389,['new_us','vol_stock_us','vol_commo_us','vix']]

exog_forecast = exog_us.loc[3390:,['new_us','vol_stock_us','vol_commo_us','vix']]


#%% Graph data
fig, axes = plt.subplots(1, 2, figsize=(10,4))
fig = sm.graphics.tsa.plot_acf(endog_train, lags=20, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(endog_train, lags=20, ax=axes[1])

#%% Fit ARMAX Model
model_us = sm.tsa.statespace.SARIMAX(endog_train, exog_train, order = (1,0,0))
result_us = model_us.fit(disp = False)
print(result_us.summary())


#%% Prediction
model_new = sm.tsa.statespace.SARIMAX(endog_test, exog_test, order = (1,0,0))
predict_us = model_new.filter(result_us.params)

predict_cor = predict_us.fittedvalues
predict_error = predict_cor - endog_test

# Graph
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
us_data.loc[3277:3389,'correlation_us'].plot(ax=ax, style='-',label='Observed')

# Plot predictions
predict_cor.loc[3277:,].plot(ax=ax, style='r--', label='One-step-ahead prediction')

legend = ax.legend(loc='best')

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(predict_cor, endog_test))
print('Test RMSE: %.3f' % rmse)

#%% Forecast by ARIMAX
nforecast = 4

new_cor = result_us.predict(start=0,end=3390+nforecast,exog=exog_forecast)

# Graph
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
us_data.loc[3277:,'correlation_us'].plot(ax=ax, style='-', label='Observed')

# Plot predictions

new_cor[3390:].plot(ax=ax, style='r--', label='One-step-ahead forecast')

legend = ax.legend(loc='best')









#%% LSTM
from numpy.random import seed
seed(4)
from tensorflow.random import set_seed
set_seed(8)

# clean dataset
from sklearn.model_selection import train_test_split
us_dataset = pd.concat([endog_us,exog_us.loc[:3389,:]],axis=1)
    
# split training set and test set
us_data_train,us_data_test = train_test_split(us_dataset,test_size = 0.01,\
                                              random_state = 1,shuffle = False)
us_data_train.head()
us_data_test.head()


#%%
# multivariate data preparation
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence for training
us_vols_train = array(us_data_train['vol_stock_us'])
us_volc_train = array(us_data_train['vol_commo_us'])
us_case_train = array(us_data_train['new_us'])
us_vix_train = array(us_data_train['vix'])
us_corr_train = array(us_data_train['correlation_us'])

# define input sequence for testing
us_vols_test = array(us_data_test['vol_stock_us'])
us_volc_test = array(us_data_test['vol_commo_us'])
us_case_test = array(us_data_test['new_us'])
us_vix_test = array(us_data_test['vix'])
us_corr_test = array(us_data_test['correlation_us'])

# convert to [rows, columns] structure
us_vols_train = us_vols_train.reshape((len(us_vols_train), 1))
us_volc_train = us_volc_train.reshape((len(us_volc_train), 1))
us_case_train = us_case_train.reshape((len(us_case_train), 1))
us_vix_train = us_vix_train.reshape((len(us_vix_train), 1))
us_corr_train = us_corr_train.reshape((len(us_corr_train), 1))

us_vols_test = us_vols_test.reshape((len(us_vols_test), 1))
us_volc_test = us_volc_test.reshape((len(us_volc_test), 1))
us_case_test = us_case_test.reshape((len(us_case_test), 1))
us_vix_test = us_vix_test.reshape((len(us_vix_test), 1))
us_corr_test = us_corr_test.reshape((len(us_corr_test), 1))

# horizontally stack columns
us_dataset_train = hstack((us_vols_train, us_volc_train,\
                           us_case_train, us_vix_train, us_corr_train))
us_dataset_test = hstack((us_vols_test, us_volc_test,\
                          us_case_test, us_vix_test, us_corr_test))

#%% choose a number of time steps
n_steps = 1
# convert into input/output
x_us_train, y_us_train = split_sequences(us_dataset_train, n_steps)
x_us_test, y_us_test = split_sequences(us_dataset_test, n_steps)

#%% the dataset knows the number of features, e.g. 2
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

n_features = x_us_train.shape[2]
# define model
n_steps = 1
model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(x_us_train, y_us_train, epochs=450, verbose=1)
    
# plot history
plt.clf()
plt.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.savefig('loss.png')
plt.close('all')
 

#%% demonstrate prediction
y_predict = model.predict(x_us_test, verbose=1)

predictions = us_data_test.drop([])

for j in range(0,33):
    for i in y_predict[j]:
        predictions.loc[3356+j:,'cor_pred'] = i
        
# GRAPH
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
predictions.loc[:,'correlation_us'].plot(ax=ax, style='-', label='Observed')

# Plot predictions
predictions.loc[:,'cor_pred'].plot(ax=ax, style='r--', label='Predicted')

legend = ax.legend(loc='best')

# calculate RMSE
rmse = sqrt(mean_squared_error(predictions.loc[:,'cor_pred'],\
                               predictions.loc[:,'correlation_us']))
print('Test RMSE: %.3f' % rmse)

#%%
# reshape exog_forecast into 3D array
x_us_forecast = np.array([[exog_forecast.loc[3390,:]],\
                          [exog_forecast.loc[3391,:]],\
                          [exog_forecast.loc[3392,:]],\
                          [exog_forecast.loc[3393,:]],\
                          [exog_forecast.loc[3394,:]]])

y_us_forecast = model.predict(x_us_forecast, verbose=1)

y_us_forecast = y_us_forecast.tolist()

new_cor2 = us_dataset.drop(['new_us','vol_stock_us','vol_commo_us','vix'],axis=1)

for j in range(0,5):
    for i in y_us_forecast[j]:
        new_cor2.loc[3390+j,'correlation_us'] = i
        
# Graph
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
us_data.loc[3277:,'correlation_us'].plot(ax=ax, style='-', label='Observed')

# Plot predictions
new_cor2.loc[3389:,'correlation_us'].plot(ax=ax, style='r--',\
            label='One-step-ahead forecast')

legend = ax.legend(loc='best')
