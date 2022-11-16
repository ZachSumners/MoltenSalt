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
# GROUPV |      |     |
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

loops = 25
for i in range(loops):
    locations = InitialConditions.InitialLocations(0, 0)
    location1 = locations[0]
    location2 = locations[1]
    radius = InitialConditions.InitialRadius(0)
    starting_coords = InitialConditions.InitialCoords(radius)
    starting_x = starting_coords[0]
    starting_y = starting_coords[1]

    
    results = DatasetConstruction.DataConstruction(location1, location2, radius, starting_x, starting_y, timelimit, rows, cols)
    calculatedtime = results[1]
    data = results[0][0]
    
    if i == 0:
        timelist = np.arange(0, len(data), 1)
    dfdata['Time Elapsed'] = timelist

    name = "Cross Correlation Sim " + str(i+1)
    dfdata[name] = data
    dfparameters[name] = np.array([radius, starting_x, starting_y, location1, location2, location2 - location1, calculatedtime])


    

dfdata.to_csv("MoltenSaltDataframe.csv")
dfparameters.to_csv("MoltenSaltParameters.csv")
