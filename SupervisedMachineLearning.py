import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#Load Cross correlation dataset
CCdata = pd.read_csv('MoltenSaltDataframe.csv')
CCdata_x = np.arange(0, CCdata.shape[0], 1)

plt.plot(CCdata_x, CCdata['Cross Correlation Sim 1'])
plt.show()

#Split data into training and testing sets
CCdata_train = CCdata[:-100]
CCdata_test = CCdata[-100:]

#Linear regression object
regr = linear_model.LinearRegression()

#Train model using training sets
#regr.fit()