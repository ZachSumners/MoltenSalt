import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, StratifiedKFold
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
    'n_estimators': np.arange(10, 1000, 10),
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2']
    }


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

pca = PCA(n_components=3)
CCdata = pca.fit_transform(CCdata)

clf = GridSearchCV(dtc, parameters)
clf.fit(CCdata, binned_CClabels)
print(clf.best_score_)

#plt.figure(figsize=(12,6))
#labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
#X_axis = np.arange(len(labels))
#ax = plt.gca()
#plt.ylim(0.40000, 1)
#plt.bar(X_axis-0.2, results['train_accuracy'], 0.4, color='blue', label='Training')
#plt.bar(X_axis+0.2, results['test_accuracy'], 0.4, color='red', label='Validation')
#plt.title('Cross Validation', fontsize=30)
#plt.xticks(X_axis, labels)
#plt.legend()
#plt.grid(True)
#plt.show()