import numpy as np
import pandas as pd
from sklearn import preprocessing
from anon_detect import anom_detect
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def sliding_window(data, window):
    current_pos = 0
    left_pos = 0
    win_size = window
    right_pos = left_pos + win_size
    vdata = []
    mdata = []
    while current_pos < len(data-win_size):
        left_pos = current_pos
        right_pos = left_pos + win_size
        mean = np.mean(np.var(data[left_pos:right_pos,:], axis=0))
        var = np.var(data[left_pos:right_pos,:], axis=0)
        vdata.append(var)
        mdata.append(mean)
        current_pos += 1
    return vdata, mdata





df = pd.read_csv('sepre.csv')
df.index.name = 'time'
df.columns = ['WTR.PSI']
numpy_matrix = df.as_matrix()

'''
an = anom_detect()
g = an.evaluate(df)
print(g)
an.normality()
'''


vv = df.rolling(window=10).mean()
vvv = vv.values.T.tolist()
#print(vvv)
dvar, dmean = sliding_window(numpy_matrix, 2)


avg = (max(dmean)+min(dmean))/2
#print(max(dmean),avg)
dd = int(avg)
for i in range(0,len(dmean)):
    if dmean[i]>dd:
        ind = dmean[i]
print(ind)

print(dmean.index(ind))


plt.plot(dmean)
plt.scatter(dmean.index(ind),ind,color='red')
plt.title('WTR.PSI')
plt.show()


'''
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
'''
