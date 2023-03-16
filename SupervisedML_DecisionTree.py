import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

#Load Cross correlation dataset
CCdata = pd.read_csv('MoltenSaltDataframeMSSolution.csv').iloc[190:310]
CCdata = CCdata.drop(['Unnamed: 0', 'Time Elapsed'], axis=1)
CCdata = CCdata.transpose()
CCdata = CCdata.to_numpy()

#scaler = MinMaxScaler()
#scaler.fit(CCdata)
#CCdata = scaler.transform(CCdata)

#Load correct labels
CClabels = pd.read_csv('MoltenSaltParametersMSSolution.csv').iloc[[6]]
CClabels = CClabels.drop(['Unnamed: 0'], axis=1)
CClabels += 198
CClabels = CClabels.astype(int)
CClabels = CClabels.to_numpy()[0]

CCdata, CClabels = shuffle(CCdata, CClabels, random_state = 45)

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

#CCdata = CCdata[:1000]
#binned_CClabels = binned_CClabels[:1000]

#CCdata = SelectKBest(chi2, k=300).fit_transform(CCdata, binned_CClabels)

#Principal Component Analysis - Reduce dimensionality before fitting.
#pca = PCA(n_components=3)
#CCdata = pca.fit_transform(CCdata)

#Classifier type
dtc = DecisionTreeClassifier()
parameters = {
    'splitter': ['random', 'best'],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': np.linspace(5, 30, 25, dtype=int),
    'min_samples_split': np.linspace(10, 50, 40, dtype=int),
    'min_samples_leaf': np.linspace(10, 50, 40, dtype=int)
    }

clf = GridSearchCV(dtc, parameters, return_train_score=True)
clf = RandomizedSearchCV(dtc, parameters, n_iter=300, return_train_score=True)
clf.fit(CCdata, binned_CClabels)

results = pd.DataFrame(clf.cv_results_)
#print(results['mean_test_score'], results['mean_train_score'])
results.to_csv('DecisionTreeFittingResults.csv')
print('Complete')