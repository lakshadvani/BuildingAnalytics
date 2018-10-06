import numpy as np
import pandas as pd
from sklearn import preprocessing
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import pacf
df = pd.read_csv('day.csv')
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df=pd.DataFrame(x_scaled, columns=df.columns)


spacetemp = df['SPACE.TEMP']
co2 = df['SPACE.CO2']
cfm = df['AVG.CFM']

fields = ['SPACE.TEMP','AVG.CFM','SPACE.CO2']
gc = pd.read_csv('day.csv', skipinitialspace=True, usecols=fields)

d = np.corrcoef(cfm,spacetemp)[0, 1]
print(d)


print(d)

from scipy import stats

print(stats.ks_2samp(co2, cfm))

spacediff = spacetemp[:-1] - spacetemp[1:]

print(spacediff)
