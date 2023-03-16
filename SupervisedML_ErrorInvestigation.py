import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier

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
clf = KNeighborsClassifier(n_neighbors=275, algorithm='ball_tree', weights='uniform')

#Principal Component Analysis - Reduce dimensionality before fitting.
pca = PCA(n_components=3)

binned_CClabels = []
for i in range(len(CClabels)):
    binned_CClabels.append(round(CClabels[i]/10)*10)
binned_CClabels = np.asarray(binned_CClabels)

badruns = np.where(binned_CClabels > 300)
CCdata = np.delete(CCdata, badruns, 0)
binned_CClabels = np.delete(binned_CClabels, badruns)

badrunsLow = np.where(binned_CClabels < 200)
CCdata = np.delete(CCdata, badrunsLow, 0)
binned_CClabels = np.delete(binned_CClabels, badrunsLow)
originalCCdata = CCdata

#counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#deletinglist = []
#for i in range(len(binned_CClabels)):
#    current = counts[int((binned_CClabels[i]-200)/10)]
#    if current < 200:
#        counts[int((binned_CClabels[i]-200)/10)] += 1
#    else:
#        deletinglist.append(i)

#CCdata = np.delete(CCdata, deletinglist, 0)
#binned_CClabels = np.delete(binned_CClabels, deletinglist, 0)

CCdata = pca.fit_transform(CCdata)

indicies = np.arange(0, len(binned_CClabels), 1)

X_train, X_test, y_train, y_test, indicies_train, indicies_test = train_test_split(CCdata, binned_CClabels, indicies, test_size=0.2)
# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


wrong = np.array([input for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])
right = np.array([input for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction == label])


wrongidx = np.array([idx[0] for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])
wronglabel = np.array([[prediction, label] for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])
rightlabel = np.array([[prediction, label] for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction == label])

rightTestidx = np.array([indicies_test[idx[0]] for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction == label])
wrongTestidx = np.array([indicies_test[idx[0]] for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])

predicted = clf.predict(X_test)

x = np.arange(0, 397, 1)
fig, ax = plt.subplots(nrows=3, ncols=3)
j = 0

for row in ax:
    for col in row:
        col.plot(x, originalCCdata[wrongTestidx[j]])
        col.set_title(wronglabel[j])
        j += 1
plt.show()

fig2, ax2 = plt.subplots(nrows=3, ncols=3)
j2 = 0

for row2 in ax2:
    for col2 in row2:
        col2.plot(x, originalCCdata[rightTestidx[j2]])
        col2.set_title(rightlabel[j2])
        j2 += 1

plt.show()

print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted, cmap=plt.cm.Blues)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()