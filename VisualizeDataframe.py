import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

df = pd.read_csv('MoltenSaltDataframe.csv')

numplots = 10

fig, axes = plt.subplots(nrows=2, ncols=5)
for i in range(numplots):
    correcttime = df['Cross Correlation Sim ' + str(i+1)][3]
    normalized = df['Cross Correlation Sim ' + str(i+1)].iloc[4:]/df['Cross Correlation Sim ' + str(i+1)].iloc[4:].abs().max()

    normalized.plot(ax=axes[i//5, i%5], title=correcttime)
    print("CROSS CORRELATION OFFSET TIME: " + str(normalized.argmax() - 98))

plt.show()