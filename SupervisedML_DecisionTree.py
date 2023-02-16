import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

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

binned_CClabels = []
for i in range(len(CClabels)):
    binned_CClabels.append(CClabels[i] - CClabels[i]%10)
binned_CClabels = np.asarray(binned_CClabels)

print(binned_CClabels)

badruns = np.where(binned_CClabels > 300)
CCdata = np.delete(CCdata, badruns, 0)
binned_CClabels = np.delete(binned_CClabels, badruns)

badrunsLow = np.where(binned_CClabels < 200)
CCdata = np.delete(CCdata, badrunsLow, 0)
binned_CClabels = np.delete(binned_CClabels, badrunsLow)

#Principal Component Analysis - Reduce dimensionality before fitting.
pca = PCA(n_components=3)
CCdata = pca.fit_transform(CCdata)

#Classifier type
dtc = DecisionTreeClassifier(splitter='best')
parameters = {
    'max_depth': np.linspace(1, 50, 50, dtype=int),
    'min_samples_split': np.linspace(1, 40, 40, dtype=int),
    'min_samples_leaf': np.linspace(1, 20, 20, dtype=int)
    }

clf = GridSearchCV(dtc, parameters, return_train_score=True)
print('...Running')
clf.fit(CCdata, binned_CClabels)


results = pd.DataFrame(clf.cv_results_)
#print(clf.cv_results_)
results.to_csv('DecisionTreeFittingResults.csv')
print('Complete')