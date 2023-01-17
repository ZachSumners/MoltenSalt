import pandas as pd
import numpy as np
from sklearn.utils import shuffle

dfdata = pd.read_csv('MoltenSaltDataframe.csv')
dfparams = pd.read_csv('MoltenSaltParameters.csv')

dfdataTranspose = dfdata.transpose()
dfparamsTranspose = dfparams.transpose()

dfdataTransposeShuffle = shuffle(dfdataTranspose, random_state=21)
dfparamsTransposeShuffle = shuffle(dfparamsTranspose, random_state =21)

dfdataShuffled = dfdataTransposeShuffle.transpose()
dfparamsShuffled = dfparamsTransposeShuffle.transpose()

dfdataShuffled.to_csv("MoltenSaltDataframe_Shuffled.csv")
dfparamsShuffled.to_csv("MoltenSaltParameters_Shuffled.csv")
