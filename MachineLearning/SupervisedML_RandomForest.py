import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

#Load Cross correlation dataset
CCdata = pd.read_csv('MoltenSaltDataframeMSSolution.csv')[200:310]
CCdata = CCdata.drop(['Unnamed: 0', 'Time Elapsed'], axis=1)
CCdata = CCdata.transpose()
CCdata = CCdata.to_numpy()

#Load correct labels
CClabels = pd.read_csv('MoltenSaltParametersMSSolution.csv').iloc[[6]]
CClabels = CClabels.drop(['Unnamed: 0'], axis=1)
CClabels += 198
CClabels = CClabels.astype(int)
CClabels = CClabels.to_numpy()[0]

CCdata, CClabels = shuffle(CCdata, CClabels, random_state = 45)

#Classifier type. Define hyperparameter bounds.
dtc = RandomForestClassifier()
parameters = {
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': np.linspace(5, 30, 25, dtype=int),
    'min_samples_split': np.linspace(10, 50, 40, dtype=int),
    'min_samples_leaf': np.linspace(10, 50, 40, dtype=int),
    'n_estimators': np.linspace(10, 2500, 250, dtype=int)
    }

#Data preprocessing. Only use labels between 200 and 300 for noise reasons.
binned_CClabels = []
for i in range(len(CClabels)):
    binned_CClabels.append(CClabels[i] - CClabels[i]%10)
binned_CClabels = np.asarray(binned_CClabels)

badruns = np.where(binned_CClabels > 300)
CCdata = np.delete(CCdata, badruns, 0)
binned_CClabels = np.delete(binned_CClabels, badruns)

badrunsLow = np.where(binned_CClabels < 200)
CCdata = np.delete(CCdata, badrunsLow, 0)
binned_CClabels = np.delete(binned_CClabels, badrunsLow)

#Hyperparameter searching and cross validation.
clf = RandomizedSearchCV(dtc, parameters, n_iter=100, return_train_score=True)
print('...Running')
clf.fit(CCdata, binned_CClabels)

#Output results.
results = pd.DataFrame(clf.cv_results_)
results.to_csv('RandomForestFittingResults.csv')
print('Complete')