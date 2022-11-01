import DatasetConstruction
import InitialConditions
import pandas as pd
import numpy as np

data = []
location1s = []
location2s = []
offsets = []
lags = []

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

timelimit = 100

loops = 2
for i in range(loops):
    location1 = InitialConditions.InitialLocations(0, 0, 50, 50, 200, 300, i)[0]
    location2 = InitialConditions.InitialLocations(0, 0, 50, 50, 200, 300, i)[1]
    radius = InitialConditions.InitialRadius(100)
    starting_x = InitialConditions.InitialCoords()[0]
    starting_y = InitialConditions.InitialCoords()[1]

    lagtime = InitialConditions.VelocityFunction(500/2)

    results = DatasetConstruction.DataConstruction(location1, location2, radius, starting_x, starting_y, timelimit)
    
    if i == 0:
        timelist = np.arange(0, len(results[0]), 1)
    dfdata['Time Elapsed'] = timelist

    name = "Cross Correlation Sim " + str(i+1)
    dfdata[name] = results[0]
    dfparameters[name] = np.array([location1, location2, abs(location2 - location1), lagtime])


    

dfdata.to_csv("MoltenSaltDataframe.csv")
dfparameters.to_csv("MoltenSaltParameters.csv")
