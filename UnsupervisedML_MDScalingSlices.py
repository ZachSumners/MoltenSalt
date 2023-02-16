import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import manifold, datasets
from matplotlib import ticker
from matplotlib.colors import ListedColormap

CCdata = pd.read_csv('MoltenSaltDataframeMSSolution.csv').loc[200:299]
CCdata = CCdata.drop(['Unnamed: 0', 'Time Elapsed'], axis=1)
CCdata = CCdata.transpose()
CCdata = CCdata.to_numpy()

slicedCCdata = CCdata.reshape(21100, 10)

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def plot_3d(points, title):
    x, y, z = points.T
    cm = ListedColormap(["#045993", "#db6000", '#118011', '#b40c0d', '#75499c', '#6d392e', '#c059a1', '#606060', '#9b9c07', '#009dad'])
    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c='red', s=50, alpha=0.8)
    ax.view_init(azim=60, elev=9)
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    #ax.zaxis.set_major_locator(ticker.MultipleLocator(1000))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()

md_scaling = manifold.MDS(
    n_components=2,
    max_iter=50,
    n_init=4,
    random_state=0
)
print('herebefore')
scaling = md_scaling.fit_transform(slicedCCdata)
print('here')

plot_3d(scaling, "Multidimensional scaling")