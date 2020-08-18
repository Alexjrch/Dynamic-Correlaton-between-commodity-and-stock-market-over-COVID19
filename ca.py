#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 19:56:24 2020

@author: alexchengjr
"""

#%%

PYTHONHASHSEED=0


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
ca_data = pd.read_csv('correlation_ca.csv')
ca_data.head()

#%%
ca_data['correlation_ca'].plot()
plt.show()

#ca_data.loc[3200:,['new_ca']].plot()
#plt.show()

#ca_data['vol_stock_ca'].plot()
#plt.show()

#ca_data['vol_commo_ca'].plot()
#plt.show()

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
    
adf_test(ca_data['correlation_ca'])

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

kpss_test(ca_data['correlation_ca'])

#%%
#ca_data['correlation_ca']=ca_data['correlation_ca'].diff()
#ca_data.drop(ca_data.head(1).index, inplace=True)
#ca_data.index = np.arange(len(ca_data))

#%% Variables
covidf_ca = pd.read_csv('forecast_ca.csv')
covidf_ca.head()
covidf_ca = covidf_ca.loc[:,['pandemic_ca_fcst','vol_stock_ca_fcst']]

covidf_ca.columns = ['pandemic','vol_stock_ca']


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

endog_ca = ca_data.loc[:,['correlation_ca']]
exog_ca = ca_data.loc[:,['pandemic','vol_stock_ca']]
exog_ca = pd.concat([exog_ca, covidf_ca], ignore_index=True, sort=False)
exog_ca = scaler.fit_transform(exog_ca)
exog_ca = pd.DataFrame(data=exog_ca[0:,0:],\
                       columns=['pandemic','vol_stock_ca'])


endog_train = endog_ca.loc[:3342,'correlation_ca']
exog_train = exog_ca.loc[:3342,['pandemic','vol_stock_ca']]

endog_test = endog_ca.loc[0:3342,'correlation_ca']
exog_test = exog_ca.loc[0:3342,['pandemic','vol_stock_ca']]


exog_forecast = exog_ca.loc[3343:,['pandemic','vol_stock_ca']]


#%% Graph data
fig, axes = plt.subplots(1, 2, figsize=(10,4))
fig = sm.graphics.tsa.plot_acf(endog_train, lags=20, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(endog_train, lags=20, ax=axes[1])

#%% Fit ARMAX Model
model_ca = sm.tsa.statespace.SARIMAX(endog_train, exog_train, order = (3,0,2))
result_ca = model_ca.fit(disp = False)
print(result_ca.summary())

#%% Prediction
model_new = sm.tsa.statespace.SARIMAX(endog_test, exog_test, order = (3,0,2))
predict_ca = model_new.filter(result_ca.params)

predict_cor = predict_ca.fittedvalues
predict_error = predict_cor - endog_test

# Graph
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
ca_data.loc[3234:3342,'correlation_ca'].plot(ax=ax, style='-',label='Observed')

# Plot predictions
predict_cor[3234:].plot(ax=ax, style='r--', label='One-step-ahead prediction')

legend = ax.legend(loc='best')

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(predict_cor, endog_test))
print('Test RMSE: %.3f' % rmse)

#%% Forecast by ARIMAX
nforecast = 39

new_cor = result_ca.predict(start=0,end=3343+nforecast,exog=exog_forecast)

# Graph
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
ca_data.loc[3000:,'correlation_ca'].plot(ax=ax, style='-', label='Observed')

# Plot predictions

new_cor[3342:].plot(ax=ax, style='r-', label='Forecast by ARIMAX(3,0,2) ')

legend = ax.legend(loc='best')









#%% LSTM
import random as python_random

np.random.seed(2)
python_random.seed(2)
tensorflow.random.set_seed(2)

#%% Variables
covidf_ca = pd.read_csv('forecast_ca.csv')
covidf_ca.head()
covidf_ca = covidf_ca.loc[:,['vol_stock_ca_fcst','pandemic_ca_fcst']]

covidf_ca.columns = ['vol_stock_ca','pandemic']


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

endog_ca = ca_data.loc[:,['correlation_ca']]
exog_ca = ca_data.loc[:,['vol_stock_ca','pandemic']]
exog_ca = pd.concat([exog_ca, covidf_ca], ignore_index=True, sort=False)
exog_ca = scaler.fit_transform(exog_ca)
exog_ca = pd.DataFrame(data=exog_ca[0:,0:],\
                       columns=['vol_stock_ca','pandemic'])


endog_train = endog_ca.loc[:3342,'correlation_ca']
exog_train = exog_ca.loc[:3342,['vol_stock_ca','pandemic']]

endog_test = endog_ca.loc[:3342,'correlation_ca']
exog_test = exog_ca.loc[:3342,['vol_stock_ca','pandemic']]


exog_forecast = exog_ca.loc[3343:,['vol_stock_ca','pandemic']]

# clean dataset
from sklearn.model_selection import train_test_split
ca_dataset = pd.concat([endog_ca,exog_ca.loc[:3342,:]],axis=1)
    
# split training set and test set
ca_data_train,ca_data_test = train_test_split(ca_dataset,test_size = 0.01,\
                                              random_state = 2,shuffle = False)
ca_data_train.head()
ca_data_test.head()


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
ca_vols_train = array(ca_data_train['vol_stock_ca'])
ca_volc_train = array(ca_data_train['pandemic'])
#ca_pand_train = array(ca_data_train['pandemic'])
ca_corr_train = array(ca_data_train['correlation_ca'])

# define input sequence for testing
ca_vols_test = array(ca_data_test['vol_stock_ca'])
ca_volc_test = array(ca_data_test['pandemic'])
#ca_pand_test = array(ca_data_test['pandemic'])
ca_corr_test = array(ca_data_test['correlation_ca'])

# convert to [rows, columns] structure
ca_vols_train = ca_vols_train.reshape((len(ca_vols_train), 1))
ca_volc_train = ca_volc_train.reshape((len(ca_volc_train), 1))
#ca_pand_train = ca_pand_train.reshape((len(ca_pand_train), 1))
ca_corr_train = ca_corr_train.reshape((len(ca_corr_train), 1))

ca_vols_test = ca_vols_test.reshape((len(ca_vols_test), 1))
ca_volc_test = ca_volc_test.reshape((len(ca_volc_test), 1))
#ca_pand_test = ca_pand_test.reshape((len(ca_pand_test), 1))
ca_corr_test = ca_corr_test.reshape((len(ca_corr_test), 1))

# horizontally stack columns
ca_dataset_train = hstack((ca_vols_train,\
                           ca_volc_train, ca_corr_train))

ca_dataset_test = hstack((ca_vols_test,\
                          ca_volc_test, ca_corr_test))


#%% choose a number of time steps
n_steps = 1
# convert into input/output
x_ca_train, y_ca_train = split_sequences(ca_dataset_train, n_steps)
x_ca_test, y_ca_test = split_sequences(ca_dataset_test, n_steps)

#%% the dataset knows the number of features, e.g. 2
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

n_features = x_ca_train.shape[2]
# define model
n_steps = 1
model = Sequential()
model.add(LSTM(10, activation='tanh', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(x_ca_train, y_ca_train, epochs=50,\
                    verbose=1, shuffle=False,validation_data=(x_ca_test,y_ca_test))
    
# plot history
plt.clf()
plt.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.savefig('loss.png')
plt.close('all')
 

#%% demonstrate prediction
y_predict = model.predict(x_ca_test, verbose=0)

predictions = ca_data_test.drop([])

for j in range(0,33):
    for i in y_predict[j]:
        predictions.loc[3309+j:,'cor_pred'] = i
        
# GRAPH
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
predictions.loc[:,'correlation_ca'].plot(ax=ax, style='-', label='Observed')

# Plot predictions
predictions.loc[:,'cor_pred'].plot(ax=ax, style='r--', label='Predicted')

legend = ax.legend(loc='best')

# calculate RMSE
rmse = sqrt(mean_squared_error(predictions.loc[:,'cor_pred'],\
                               predictions.loc[:,'correlation_ca']))
print('Test RMSE: %.3f' % rmse)

#%%
# reshape exog_forecast into 3D array
exog_forecast = exog_forecast.loc[:,['vol_stock_ca','pandemic']]
x_ca_forecast = np.array([[exog_forecast.loc[3343,:]],\
                          [exog_forecast.loc[3344,:]],\
                          [exog_forecast.loc[3345,:]],\
                          [exog_forecast.loc[3346,:]],\
                          [exog_forecast.loc[3347,:]],\
                          [exog_forecast.loc[3348,:]],\
                          [exog_forecast.loc[3349,:]],\
                          [exog_forecast.loc[3350,:]],\
                          [exog_forecast.loc[3351,:]],\
                          [exog_forecast.loc[3352,:]],\
                          [exog_forecast.loc[3353,:]],\
                          [exog_forecast.loc[3354,:]],\
                          [exog_forecast.loc[3355,:]],\
                          [exog_forecast.loc[3356,:]],\
                          [exog_forecast.loc[3357,:]],\
                          [exog_forecast.loc[3358,:]],\
                          [exog_forecast.loc[3359,:]],\
                          [exog_forecast.loc[3360,:]],\
                          [exog_forecast.loc[3361,:]],\
                          [exog_forecast.loc[3362,:]],\
                          [exog_forecast.loc[3363,:]],\
                          [exog_forecast.loc[3364,:]],\
                          [exog_forecast.loc[3365,:]],\
                          [exog_forecast.loc[3366,:]],\
                          [exog_forecast.loc[3367,:]],\
                          [exog_forecast.loc[3368,:]],\
                          [exog_forecast.loc[3369,:]],\
                          [exog_forecast.loc[3370,:]],\
                          [exog_forecast.loc[3371,:]],\
                          [exog_forecast.loc[3372,:]],\
                          [exog_forecast.loc[3373,:]],\
                          [exog_forecast.loc[3374,:]],\
                          [exog_forecast.loc[3375,:]],\
                          [exog_forecast.loc[3376,:]],\
                          [exog_forecast.loc[3377,:]],\
                          [exog_forecast.loc[3378,:]],\
                          [exog_forecast.loc[3379,:]],\
                          [exog_forecast.loc[3380,:]],\
                          [exog_forecast.loc[3381,:]],\
                          [exog_forecast.loc[3382,:]]
                          ])

    
    
y_ca_forecast = model.predict(x_ca_forecast, verbose=0)

y_ca_forecast = y_ca_forecast.tolist()

new_cor2 = ca_dataset.drop(['vol_stock_ca','pandemic'],axis=1)

for j in range(0,40):
    for i in y_ca_forecast[j]:
        new_cor2.loc[3343+j,'correlation_ca'] = i
        
# Graph
fig, ax = plt.subplots(figsize=(9,6))
npre = 4
ax.set(title='', xlabel='', ylabel='')

# Plot data points
ca_data.loc[3000:,'correlation_ca'].plot(ax=ax, style='-', label='Observed')

# Plot predictions
new_cor2.loc[3342:,'correlation_ca'].plot(ax=ax, style='r-',label='Forecast by LSTM')

legend = ax.legend(loc='best')





