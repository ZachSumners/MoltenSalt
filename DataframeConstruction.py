import DatasetConstruction
import pandas as pd
import numpy as np

data = []
location1s = []
location2s = []
offsets = []

loops = 5
for i in range(loops):
    location1 = 200#50*i + 150
    location2 = 50*i + 250
    results = DatasetConstruction.DataConstruction(location1, location2)

    data.append(results[0][0])
    location1s.append(results[0][1])
    location2s.append(results[0][2])
    offsets.append(results[0][3])

dfdata = pd.DataFrame()

#############################################
# TIME ELAPSED | SIM1   | SIM2 | ... | SIM N
# -1           | LOC1   |      |     |
# -1           | LOC2   |      |     |
# -1           | OFFSET |      |     |
# 0            | DATA   |      |     |
# ...          | DATA   |      |     |
# END          | DATA   |      |     |
#############################################

timelist = np.arange(0, len(data[0]), 1)
timelist = np.insert(timelist, 0, np.array([-1, -1, -1]))
dfdata['Time Elapsed'] = timelist
for i in range(loops):
    name = "Cross Correlation Sim " + str(i+1)
    data[i].insert(0, location1s[i])
    data[i].insert(1, location2s[i])
    data[i].insert(2, offsets[i])
    dfdata[name] = data[i]

dfdata.to_csv("MoltenSaltDataframe.csv")
