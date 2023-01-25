import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC

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
dtc = SVC()
parameters = {
    'C': np.linspace(0.01, 5, 10),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    #'degree': np.linspace(1, 5, 5, dtype=int),
    'gamma': ['scale', 'auto']
    }
clf = GridSearchCV(dtc, parameters)

#Principal Component Analysis - Reduce dimensionality before fitting.
pca = PCA(n_components=3)

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

#Split data into testing and training.
#X_train, X_test, y_train, y_test = train_test_split(CCdata, binned_CClabels, test_size=0.1, shuffle=True, stratify=binned_CClabels)
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(CCdata, binned_CClabels)

cv = []

for i, (train_index, test_index) in enumerate(skf.split(CCdata, binned_CClabels)):
    print(test_index)
    X_train, X_test, y_train, y_test = CCdata[train_index], CCdata[test_index], binned_CClabels[train_index], binned_CClabels[test_index]

    #counts, bins = np.histogram(y_train, bins=np.arange(195, 326, 5))
    #plt.bar(bins[:-1], counts, width = 4)
    #countsTest, binsTest = np.histogram(y_test, bins=np.arange(195, 326, 5))
    #plt.bar(binsTest[:-1], countsTest, width = 4)
    #plt.show()

    pca.fit(X_train)
    X_test = pca.transform(X_test)
    X_train = pca.transform(X_train)
    #y_train = pca.transform(y_train)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)


    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    cv.append(metrics.accuracy_score(y_test, predicted))

    #disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted, cmap=plt.cm.Blues)
    #disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    #plt.show()

plt.bar([1, 2, 3, 4, 5], cv)
print(cv)
plt.show()