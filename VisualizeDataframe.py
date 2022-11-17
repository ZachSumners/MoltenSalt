import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal

df = pd.read_csv('MoltenSaltDataframe.csv')
dfparams = pd.read_csv('MoltenSaltParameters.csv')

numplots = len(df.columns) - 2
averagecount = 0
avgcount = 0
fig, axes = plt.subplots(nrows=2, ncols=2)
for i in range(numplots):
    correcttime = round(dfparams['Cross Correlation Sim ' + str(i+1)][6], 2)
    normalized = df['Cross Correlation Sim ' + str(i+1)]/df['Cross Correlation Sim ' + str(i+1)].abs().max()

    #ax = df.plot('Time Elapsed', 'Normalized', label='Cross Correlation', title = 'Cross Correlation of Two Signals')
    ax = normalized.plot(ax=axes[i//5, i%5], title="Cross Correlation of Two Signals")#, title=correcttime+198)
    ax.set(xlabel = 'Lag Time', ylabel = 'Normalized Strength')
    lags = scipy.signal.correlation_lags((len(normalized)+1)/2, (len(normalized)+1)/2, mode='full')
    lag = lags[np.argmax(normalized)]

    print("------------------------------")
    print("MEASURED VELOCITY: ", dfparams['Cross Correlation Sim ' + str(i+1)][5]/lag)
    print("CALCULATED VELOCITY: ", dfparams['Cross Correlation Sim ' + str(i+1)][5]/dfparams['Cross Correlation Sim ' + str(i+1)][6])
    print("ERROR: ", abs(dfparams['Cross Correlation Sim ' + str(i+1)][5]/lag - dfparams['Cross Correlation Sim ' + str(i+1)][5]/dfparams['Cross Correlation Sim ' + str(i+1)][6]))
    print("------------------------------")
    averagecount += dfparams['Cross Correlation Sim ' + str(i+1)][5]/lag
    avgcount += dfparams['Cross Correlation Sim ' + str(i+1)][5]/dfparams['Cross Correlation Sim ' + str(i+1)][6]
print("AVERAGE MEASURED VELOCITY: ", averagecount/numplots)
print("AVERAGE CALCULATED VELOCITY: ", avgcount/numplots)
plt.show()