import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal

idx = 196

df = pd.read_csv('MoltenSaltDataframeMSSolution.csv')
dfparams = pd.read_csv('MoltenSaltParametersMSSolution.csv')

correcttime = round(dfparams['Cross Correlation Sim ' + str(idx)][6], 2)
normalized = df['Cross Correlation Sim ' + str(idx)]#/df['Cross Correlation Sim ' + str(idx)].abs().max()
#fig = plt.figure(figsize=(10, 4))
plt.plot(np.arange(-198, 199, 1), normalized)

lags = scipy.signal.correlation_lags((len(normalized)+1)/2, (len(normalized)+1)/2, mode='full')
lag = lags[np.argmax(normalized)]
plt.title('Cross-Correlation of Two Signals')
plt.xlabel('Lag Time')
plt.ylim(-10000, 60000)
plt.ylabel('Strength')
plt.grid()

print("------------------------------")
print("MEASURED VELOCITY: ", dfparams['Cross Correlation Sim ' + str(idx)][5]/lag)
print("CALCULATED VELOCITY: ", dfparams['Cross Correlation Sim ' + str(idx)][5]/dfparams['Cross Correlation Sim ' + str(idx)][6])
print("ERROR: ", abs(dfparams['Cross Correlation Sim ' + str(idx)][5]/lag - dfparams['Cross Correlation Sim ' + str(idx)][5]/dfparams['Cross Correlation Sim ' + str(idx)][6]))
print("------------------------------")

plt.show()