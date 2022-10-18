import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('MoltenSaltDataframe.csv')

numplots = 5
for i in range(numplots):
    df.iloc[3:].plot(x = 'Time Elapsed', y = 'Cross Correlation Sim ' + str(i+1), kind='line')

plt.show()