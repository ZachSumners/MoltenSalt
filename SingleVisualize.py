import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal

idx = 1829

df = pd.read_csv('MoltenSaltDataframe.csv')
dfparams = pd.read_csv('MoltenSaltParameters.csv')

correcttime = round(dfparams['Cross Correlation Sim ' + str(idx)][6], 2)
normalized = df['Cross Correlation Sim ' + str(idx)]/df['Cross Correlation Sim ' + str(idx)].abs().max()
ax = normalized.plot()
lags = scipy.signal.correlation_lags((len(normalized)+1)/2, (len(normalized)+1)/2, mode='full')
lag = lags[np.argmax(normalized)]

print("------------------------------")
print("MEASURED VELOCITY: ", dfparams['Cross Correlation Sim ' + str(idx)][5]/lag)
print("CALCULATED VELOCITY: ", dfparams['Cross Correlation Sim ' + str(idx)][5]/dfparams['Cross Correlation Sim ' + str(idx)][6])
print("ERROR: ", abs(dfparams['Cross Correlation Sim ' + str(idx)][5]/lag - dfparams['Cross Correlation Sim ' + str(idx)][5]/dfparams['Cross Correlation Sim ' + str(idx)][6]))
print("------------------------------")

plt.show()