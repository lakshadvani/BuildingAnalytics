import numpy as np
import pandas as pd
from sklearn import preprocessing
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import pacf
df = pd.read_csv('1b25ex.csv')
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df=pd.DataFrame(x_scaled, columns=df.columns)


spacetemp = df['SPACE.TEMP']
co2 = df['SPACE.CO2']
cfm = df['AVG.CFM']
fields = ['SPACE.TEMP', 'SPACE.CO2']
gc = pd.read_csv('1b25ex.csv', skipinitialspace=True, usecols=fields)

d = np.corrcoef(co2,cfm)[0, 1]
print(d)

from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(cfm)
pyplot.title('cfm - 1b25')
pyplot.show()
d = []
for i in range(len(co2)):
    d.append(i)
df.index = pd.to_datetime(df.index, unit='ms')

co2 = pd.to_datetime(df.index)
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
model = ARIMA(co2, order=(1000,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
