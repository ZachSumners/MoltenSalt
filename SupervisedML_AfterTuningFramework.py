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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import validation_curve
from scipy.interpolate import make_interp_spline
from sklearn.svm import SVC

#Load Cross correlation dataset
CCdata = pd.read_csv('MoltenSaltDataframeMSSolution.csv')#[:2000]
CCdata = CCdata.drop(['Time Elapsed'], axis=1)
CCdata = CCdata.transpose()
CCdata = CCdata.to_numpy()

#Load correct labels
CClabels = pd.read_csv('MoltenSaltParametersMSSolution.csv').iloc[[6]]#[:2000]
#CClabels = CClabels.drop(['Unnamed: 0'], axis=1)
CClabels += 198
CClabels = CClabels.astype(int)
CClabels = CClabels.to_numpy()[0]



#Classifier type
#clf = DecisionTreeClassifier(splitter='random', max_depth= 18, min_samples_leaf= 24, min_samples_split= 40, criterion='gini')
#clf = RandomForestClassifier(n_estimators=1390, min_samples_split=32, min_samples_leaf=35, max_depth=30, criterion='log_loss', bootstrap=True)
#clf = KNeighborsClassifier(weights='uniform', n_neighbors=100, algorithm='kd_tree')
clf = SVC(C=0.31152, gamma='scale', kernel='rbf')
#clf = SVC(C=0.2152, degree=2)

#Principal Component Analysis - Reduce dimensionality before fitting.
#pca = PCA(n_components=3)
CCdata, CClabels = shuffle(CCdata, CClabels, random_state = 21)
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

#weightedF1 = []
#for i in range(1):
    #print(i)
    
    #cutCCdata = CCdata#[:i*50+500]
    #cutbinned_CClabels = binned_CClabels#[:i*50+500]

    #print(binned_CClabels)
    #CCdata = pca.fit_transform(CCdata)

    #score = cross_val_score(clf, CCdata, binned_CClabels ,cv=5)
    #print(score)

    #Split data into testing and training.
    #reportAvgList = []
    #for j in range(1):
        #print(j)
X_train, X_test, y_train, y_test = train_test_split(CCdata, binned_CClabels, test_size=0.2, shuffle=True, stratify=binned_CClabels)

        # Learn the digits on the train subset
clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

        #report = metrics.precision_recall_fscore_support(y_test, predicted, average='weighted')
        #reportAvgList.append(report[2])
    
    #reportAvg = sum(reportAvgList)/10
    #weightedF1.append(reportAvg)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["0", '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], cmap=plt.cm.Blues)
disp.figure_.suptitle("Linear SVM Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")
#X_Y_Spline = make_interp_spline(np.arange(0, 40, 1), weightedF1)
 
# Returns evenly spaced numbers
# over a specified interval.
#X_ = np.linspace(0, 40, 100)
#Y_ = X_Y_Spline(X_)
#plt.show()
#plt.plot(X_, Y_)
#plt.ylim(0, 1)
plt.show()
