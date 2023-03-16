import pandas as pd
import numpy as np
from sklearn.utils import shuffle

dfdata = pd.read_csv('MoltenSaltDataframeMSSolution.csv')
dfparams = pd.read_csv('MoltenSaltParametersMSSolution.csv')

dfdataShuffled, dfparamsShuffled = shuffle(dfdata, dfparams, random_state = 21)

dfdataShuffled.to_csv("MoltenSaltDataframe_Shuffled.csv")
dfparamsShuffled.to_csv("MoltenSaltParameters_Shuffled.csv")
