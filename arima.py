import pandas as pd
import numpy as np
#data = pd.read_csv('day.csv',index_col=0,parse_dates=[0])
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA


data = pd.read_csv('day.csv',index_col=0,parse_dates=[0])
hold = pd.Series(data['SPACE.CO2'])
spacetemp = data['SPACE.TEMP']
co2 = data['SPACE.CO2']
cfm = data['AVG.CFM']
data.index = pd.to_datetime(data.index)
print(data.index)
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(hold, model='additive', freq = 10)

result.plot()
pyplot.show()

from sklearn.metrics import mean_squared_error
series = Series.from_csv('ex.csv', header=0)
series.plot()
pyplot.show()
result = seasonal_decompose(series, model='additive',freq = 144)
result.plot()
pyplot.show()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
'''
model = ARIMA(series.astype(float), order=(5,0,1))

model_fit = model.fit(disp=1)
print(model_fit.summary())
'''
import math

X = series.astype(float).values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = np.around(output[0])
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
t = accuracy_score(test, predictions)
print("ajax",t)
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
'''

series = pd.read_csv('day.csv',index_col=0,parse_dates=[0])

series.columns = ['Date','SPACE.CO2','SPACE.TEMP','AVG.CFM','CFM.MAX','CFM.MIN','CFM.SETPT','COOL.SETPT','HEAT','HEAT.CFM','HEAT.SETPT','OCCUPIED','SP.OCC.CLG','SP.OCC.HTG','SP.SPACE.TEMP','SP.UNOCC.CLG','SP.UNOCC.HTG','WARMUPDAMPER']
dates=series['Date']
print(dates)
model = ARIMA(series['SPACE.CO2'].astype(float), order=(2,0,1), dates=series['Date'])
'''
