import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
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
dtc = KNeighborsClassifier()
parameters = {
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    }
clf = GridSearchCV(dtc, parameters)

#Principal Component Analysis - Reduce dimensionality before fitting.
pca = PCA(n_components=2)

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
originalCCdata = CCdata
CCdata = pca.fit_transform(CCdata)

#Split data into testing and training.
#X_train, X_test, y_train, y_test = train_test_split(CCdata, binned_CClabels, test_size=0.1, shuffle=True, stratify=binned_CClabels)
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(CCdata, binned_CClabels)

cv = []

for i, (train_index, test_index) in enumerate(skf.split(CCdata, binned_CClabels)):

    
    

    X_train, X_test, y_train, y_test = CCdata[train_index], CCdata[test_index], binned_CClabels[train_index], binned_CClabels[test_index]

    #counts, bins = np.histogram(y_train, bins=np.arange(195, 326, 5))
    #plt.bar(bins[:-1], counts, width = 4)
    #countsTest, binsTest = np.histogram(y_test, bins=np.arange(195, 326, 5))
    #plt.bar(binsTest[:-1], countsTest, width = 4)
    #plt.show()

    #pca.fit(X_train)
    #X_test = pca.transform(X_test)
    #X_train = pca.transform(X_train)
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



    #for idx, input, prediction, label in zip(enumerate(X_test), X_test, predicted, y_test):
    #    if prediction != label:
    #        print(idx[0])

    wrong = np.array([input for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])
    right = np.array([input for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction == label])
    wrongidx = np.array([idx[0] for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])
    wronglabel = np.array([[prediction, label] for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])
    
    rightTestidx = np.array([(y_test[idx[0]]) for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction == label])
    wrongTestidx = np.array([(y_test[idx[0]]) for idx,input,prediction,label in zip(enumerate(X_test), X_test, predicted, y_test) if prediction != label])

    cm = ListedColormap(["#045993", "#db6000", '#118011', '#b40c0d', '#75499c', '#6d392e', '#c059a1', '#606060', '#9b9c07', '#009dad'])

    fig, (ax, axs2) = plt.subplots(2)
    #print(clf)
    #boundaries = DecisionBoundaryDisplay.from_estimator(clf, CCdata, cmap=cm, grid_resolution = 100, alpha=1, ax=ax, eps=1)
    
    #print(boundaries.surface_.levels)
    #ax.set_title("Input data")

    # Plot the training points
    #ax.scatter(wrong[:, 0], wrong[:, 1], c=wrongTestidx, cmap=cm, alpha=1, edgecolors="r")
    
    # Plot the testing points
    #ax.scatter(right[:, 0], right[:, 1], c=rightTestidx, cmap=cm, alpha=1, edgecolors="k")
    
    #handles, labels = scatter.legend_elements()
    #ax.legend(handles=handles, labels=["200", '210', '220', '230', '240', '250', '260', '270', '280', '290', '300'], title="true class")
    #ax.set_xticks(())
    #ax.set_yticks(())

    h = 1000
    x_min, x_max = CCdata[:, 0].min() - 1, CCdata[:, 0].max() + 1
    y_min, y_max = CCdata[:, 1].min() - 1, CCdata[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cm)

    # Plot also the training points
     # Plot the training points
    ax.scatter(wrong[:, 0], wrong[:, 1], c=wrongTestidx, cmap=cm, alpha=1, edgecolors="r")
    
    # Plot the testing points
    ax.scatter(right[:, 0], right[:, 1], c=rightTestidx, cmap=cm, alpha=1, edgecolors="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


    
    wrongchoice = 0
    plotindex = wrongidx[wrongchoice]

    ax.scatter(wrong[plotindex, 0], wrong[plotindex, 1], c='red', edgecolors='green')

    axs2.axvline(x=wronglabel[wrongchoice][0], label='Predicted', linestyle='dashed', c='black')
    axs2.axvline(x=wronglabel[wrongchoice][1], label='Correct', linestyle='dashed', c='red')
    axs2.set_title(str(wronglabel[wrongchoice]))
    axs2.legend()
    axs2.plot(np.linspace(0, len(originalCCdata[test_index[plotindex]]), len(originalCCdata[test_index[plotindex]])), originalCCdata[test_index[plotindex]])
    plt.show()
    #disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted, cmap=plt.cm.Blues)
    #disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    #plt.show()



plt.bar([1, 2, 3, 4, 5], cv)
print(cv)
plt.show()