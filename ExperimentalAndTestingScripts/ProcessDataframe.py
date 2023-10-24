import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    binned_CClabels.append(round(CClabels[i]/10)*10)
binned_CClabels = np.asarray(binned_CClabels)

badruns = np.where(binned_CClabels > 300)
CCdata = np.delete(CCdata, badruns, 0)
binned_CClabels = np.delete(binned_CClabels, badruns)

badrunsLow = np.where(binned_CClabels < 200)
CCdata = np.delete(CCdata, badrunsLow, 0)
binned_CClabels = np.delete(binned_CClabels, badrunsLow)

counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
deletinglist = []
for i in range(len(binned_CClabels)):
    current = counts[int((binned_CClabels[i]-200)/10)]
    if current < 200:
        counts[int((binned_CClabels[i]-200)/10)] += 1
    else:
        deletinglist.append(i)

CCdata = np.delete(CCdata, deletinglist, 0)
binned_CClabels = np.delete(binned_CClabels, deletinglist, 0)