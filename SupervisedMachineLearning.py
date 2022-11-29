import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#Load Cross correlation dataset
CCdata = pd.read_csv('MoltenSaltDataframe.csv')
print(CCdata.shape)

rng = np.random.RandomState(1)
X = np.sort(396 * rng.rand(397, 1), axis=0)


#Split data into training and testing sets
regr_1 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, CCdata)

X_test = np.arange(0.0, 396, 1)[:, np.newaxis]
y_1 = regr_1.predict(X_test)

plt.scatter(X, CCdata, s=10, c='red', label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()