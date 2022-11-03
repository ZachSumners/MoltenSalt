import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal

df = pd.read_csv('MoltenSaltDataframe.csv')
dfparams = pd.read_csv('MoltenSaltParameters.csv')

numplots = len(df.columns) - 2

fig, axes = plt.subplots(nrows=2, ncols=5)
for i in range(numplots):
    correcttime = round(dfparams['Cross Correlation Sim ' + str(i+1)][6], 2)
    normalized = df['Cross Correlation Sim ' + str(i+1)]/df['Cross Correlation Sim ' + str(i+1)].abs().max()

    normalized.plot(ax=axes[i//5, i%5], title=correcttime)
    lags = scipy.signal.correlation_lags(len(normalized)/2, len(normalized)/2, mode='full')
    lag = lags[np.argmax(normalized)]
    velocity = dfparams['Cross Correlation Sim ' + str(i+1)][5]/lag
    print("CROSS CORRELATION OFFSET TIME: ", velocity)

plt.show()