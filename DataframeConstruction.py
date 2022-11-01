import DatasetConstruction
import InitialConditions
import pandas as pd
import numpy as np

data = []
location1s = []
location2s = []
offsets = []
lags = []

loops = 5
for i in range(loops):
    location1 = InitialConditions.InitialLocations()[0]
    location2 = InitialConditions.InitialLocations()[1]
    radius = InitialConditions.InitialRadius()
    results = DatasetConstruction.DataConstruction(location1, location2, radius)

    data.append(results[0][0])
    location1s.append(results[0][1])
    location2s.append(results[0][2])
    offsets.append(results[0][3])
    lags.append(results[0][4])

dfparameters = pd.DataFrame()

#############################################
# SIM1   | SIM2 | ... | SIM N
# LOC1   |      |     |
# LOC2   |      |     |
# RADIUS |      |     |
# OFFSET |      |     |
# LAG    |      |     |
#############################################

dfdata = pd.DataFrame()

#############################################
# TIME ELAPSED | SIM1   | SIM2 | ... | SIM N
# 0            | DATA   |      |     |
# ...          | DATA   |      |     |
# END          | DATA   |      |     |
#############################################

timelist = np.arange(0, len(data[0]), 1)
dfdata['Time Elapsed'] = timelist

for i in range(loops):
    name = "Cross Correlation Sim " + str(i+1)
    dfdata[name] = data[i]
    dfparameters[name] = np.array([location1s[i], location2s[i], offsets[i], lags[i]])

dfdata.to_csv("MoltenSaltDataframe.csv")
dfparameters.to_csv("MoltenSaltParameters.csv")
