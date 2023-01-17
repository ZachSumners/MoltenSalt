import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

#Load Cross correlation dataset
CCdata = pd.read_csv('./DatasetBackup/MoltenSaltDataframe500.csv')
CCdata = CCdata.drop(['Unnamed: 0', 'Time Elapsed'], axis=1)
CCdata = CCdata.transpose()
CCdata = CCdata.to_numpy()

#Load correct labels
CClabels = pd.read_csv('./DatasetBackup/MoltenSaltParameters500.csv').iloc[[6]]
CClabels = CClabels.drop(['Unnamed: 0'], axis=1)
CClabels += CCdata.shape[1]/2
CClabels = CClabels.astype(int)
CClabels = CClabels.to_numpy()[0]

#Classifier type
clf = DecisionTreeClassifier(criterion='entropy')

#Principal Component Analysis - Reduce dimensionality before fitting.
pca = PCA(n_components=3)

#Split data into testing and training.

X_train, X_test, y_train, y_test = train_test_split(CCdata, CClabels, test_size=0.2, shuffle=True)



binned_y_train = []
binned_y_test = []
for i in range(len(y_train)):
    binned_y_train.append(y_train[i] - y_train[i]%10)
for i in range(len(y_test)):
    binned_y_test.append(y_test[i] - y_test[i]%10) 

np.asarray(binned_y_test)
np.asarray(binned_y_train)

counts, bins = np.histogram(y_train, bins=np.arange(195, 326, 5))
countsTest, binsTest = np.histogram(y_test, bins=np.arange(195, 326, 5))
plt.bar(bins[:-1], counts, width = 4)
plt.bar(binsTest[:-1], countsTest, width = 4)

print(binned_y_train)
print(binned_y_test)

pca.fit(X_train)
X_test = pca.transform(X_test)
X_train = pca.transform(X_train)
#y_train = pca.transform(y_train)

# Learn the digits on the train subset
clf.fit(X_train, binned_y_train)
print(clf.n_classes_)
print(clf.get_n_leaves())



# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(binned_y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(binned_y_test, predicted, cmap=plt.cm.Blues)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

tree.plot_tree(clf, max_depth = 2, fontsize = 10)
plt.show()
