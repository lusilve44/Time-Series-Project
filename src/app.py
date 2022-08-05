import pandas as pd
from datetime import datetime
import numpy as np
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

cpu_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv')
cpu_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv')

cpu_train_a['datetime'] = pd.to_datetime(cpu_train_a['datetime'])
cpu_train_b['datetime'] = pd.to_datetime(cpu_train_b['datetime'])

cpu_train_a = cpu_train_a.set_index('datetime')
cpu_train_b = cpu_train_b.set_index('datetime')


# model a
stepwise_model_a = auto_arima(cpu_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

stepwise_model_a.fit(cpu_train_a)

# model b
stepwise_model_b = auto_arima(cpu_train_b, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

stepwise_model_b.fit(cpu_train_b)
                           
                        


# Saving the models
filename_a = '/workspace/Time-Series-Project/models/model_a.sav'
pickle.dump(stepwise_model_a, open(filename_a, 'wb'))

filename_b = '/workspace/Time-Series-Project/models/model_b.sav'
pickle.dump(stepwise_model_b, open(filename_b, 'wb'))