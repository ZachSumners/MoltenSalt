import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal

#Investigation of the group velocity discrepancy.

df = pd.read_csv('MoltenSaltDataframe.csv')
dfparams = pd.read_csv('MoltenSaltParameters.csv')

errorlist = []
offsetlist = []
xvals = []
deformationlist = []

numplots = len(df.columns) - 2
for i in range(numplots):
    normalized = df['Cross Correlation Sim ' + str(i+1)]/df['Cross Correlation Sim ' + str(i+1)].abs().max()
    lags = scipy.signal.correlation_lags((len(normalized)+1)/2, (len(normalized)+1)/2, mode='full')
    lag = lags[np.argmax(normalized)]

    error = (dfparams['Cross Correlation Sim ' + str(i+1)][5]/lag - dfparams['Cross Correlation Sim ' + str(i+1)][5]/dfparams['Cross Correlation Sim ' + str(i+1)][6])
    offset = dfparams['Cross Correlation Sim ' + str(i+1)][5]
    xstart = dfparams['Cross Correlation Sim ' + str(i+1)][1]
    deformation = dfparams['Cross Correlation Sim ' + str(i+1)][7]
    
    if error > -2:
        errorlist.append(error)
        offsetlist.append(offset)
        xvals.append(xstart)
        deformationlist.append(deformation)

#xvals = np.asarray(xvals)
#errorlist = np.asarray(errorlist)
#idx = np.isfinite(xvals) & np.isfinite(errorlist) 
#x1, x0 = np.polyfit(xvals[idx], errorlist[idx], 1)
#print(x1, x0)

#x = np.arange(0, 500, 1)
plt.scatter(xvals, deformationlist)
#plt.scatter(deformationlist, errorlist)
#plt.plot(x, x1 * x + x0)
#plt.plot(x, 0.05*x)
#plt.plot(x, 0.0001*x**2 - 0.05*x + 6.75)
plt.xlabel('Median X Position')
plt.ylabel('Velocity Error')
#plt.ylim(-0.1, 2)
plt.xlim(0, 500)
plt.grid()
plt.show()