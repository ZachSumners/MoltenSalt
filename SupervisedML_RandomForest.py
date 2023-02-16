import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_validate, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Load Cross correlation dataset
CCdata = pd.read_csv('MoltenSaltDataframeMSSolution.csv')
CCdata = CCdata.drop(['Unnamed: 0', 'Time Elapsed'], axis=1)
CCdata = CCdata.transpose()
CCdata = CCdata.to_numpy()

#Load correct labels
CClabels = pd.read_csv('MoltenSaltParametersMSSolution.csv').iloc[[6]]
CClabels = CClabels.drop(['Unnamed: 0'], axis=1)
CClabels += CCdata.shape[1]/2
CClabels = CClabels.astype(int)
CClabels = CClabels.to_numpy()[0]

#Classifier type
dtc = RandomForestClassifier()
parameters = {
    'max_depth': np.linspace(3, 15, 13, dtype=int),
    'bootstrap': [True, False],
    'min_samples_leaf': np.linspace(2, 20, 10, dtype=int),
    'min_samples_split': np.linspace(2, 50, 48, dtype=int),
    'n_estimators': np.linspace(100, 2000, 190, dtype=int)
    }


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

pca = PCA(n_components=3)
CCdata = pca.fit_transform(CCdata)

clf = RandomizedSearchCV(dtc, parameters, n_iter=100, return_train_score=True)
print('...Running')
clf.fit(CCdata, binned_CClabels)

results = pd.DataFrame(clf.cv_results_)
results.to_csv('RandomForestFittingResults.csv')
print('Complete')