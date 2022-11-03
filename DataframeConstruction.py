import DatasetConstruction
import InitialConditions
import pandas as pd
import numpy as np
import math

dfparameters = pd.DataFrame()

#############################################
# SIM1   | SIM2 | ... | SIM N
# RADIUS |      |     |
# X0     |      |     |
# Y0     |      |     |
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

timelimit = 200
rows = 500
cols = 2000

loops = 2
for i in range(loops):
    location1 = InitialConditions.InitialLocations(0, 0)[0]
    location2 = InitialConditions.InitialLocations(0, 0)[1]
    radius = InitialConditions.InitialRadius(0)
    starting_x = InitialConditions.InitialCoords()[0]
    starting_y = InitialConditions.InitialCoords()[1]

    velocity = math.floor(InitialConditions.VelocityFunction(rows/2, rows))

    results = DatasetConstruction.DataConstruction(location1, location2, radius, starting_x, starting_y, timelimit, rows, cols)
    
    if i == 0:
        timelist = np.arange(0, len(results[0]), 1)
    dfdata['Time Elapsed'] = timelist

    name = "Cross Correlation Sim " + str(i+1)
    dfdata[name] = results[0]
    dfparameters[name] = np.array([radius, starting_x, starting_y, location1, location2, location2 - location1, velocity])


    

dfdata.to_csv("MoltenSaltDataframe.csv")
dfparameters.to_csv("MoltenSaltParameters.csv")
