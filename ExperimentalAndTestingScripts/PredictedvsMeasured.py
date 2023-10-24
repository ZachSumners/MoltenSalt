import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal

df = pd.read_csv('MoltenSaltDataframe.csv')
dfparams = pd.read_csv('MoltenSaltParameters.csv')

measuredlist = []
predictedlist = []

numplots = len(df.columns) - 2
for i in range(numplots):
    normalized = df['Cross Correlation Sim ' + str(i+1)]/df['Cross Correlation Sim ' + str(i+1)].abs().max()
    lags = scipy.signal.correlation_lags((len(normalized)+1)/2, (len(normalized)+1)/2, mode='full')
    lag = lags[np.argmax(normalized)]

    predicted = dfparams['Cross Correlation Sim ' + str(i+1)][5]/dfparams['Cross Correlation Sim ' + str(i+1)][6]
    measured = dfparams['Cross Correlation Sim ' + str(i+1)][5]/lag

    if abs(predicted - measured) < 2:
        measuredlist.append(measured)
        predictedlist.append(predicted)

    

plt.scatter(measuredlist, predictedlist)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.xlim(10, 13)
plt.ylim(10, 13)
plt.grid()
plt.show()