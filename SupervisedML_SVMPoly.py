import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import shuffle

#Load Cross correlation dataset
CCdata = pd.read_csv('MoltenSaltDataframeMSSolution.csv').iloc[200:310]
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

#Classifier type
dtc = SVC(kernel='poly')
parameters = {
    'C': np.linspace(0.1, 2, 100),
    'degree': np.linspace(1, 4, 4, dtype=int)
    #'gamma': np.logspace(-9, 3, 13)
    }

#Principal Component Analysis - Reduce dimensionality before fitting.
#pca = PCA(n_components=3)
#CCdata = pca.fit_transform(CCdata)

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

clf = GridSearchCV(dtc, parameters, return_train_score=True)
#clf = RandomizedSearchCV(dtc, parameters, n_iter = 100, return_train_score=True)
print('...Running')
clf.fit(CCdata, binned_CClabels)

results = pd.DataFrame(clf.cv_results_)
results.to_csv('SVMPolyFittingResults.csv')
print('Complete')