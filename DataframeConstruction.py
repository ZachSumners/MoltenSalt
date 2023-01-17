#import DatasetConstruction
import InitialConditions
import DatasetConstructionNonVisual
import pandas as pd
import numpy as np
import math
from timeit import default_timer as timer

concatparameters = pd.DataFrame()

#############################################
# SIM1   | SIM2 | ... | SIM N
# RADIUS |      |     |
# X0     |      |     |
# Y0     |      |     |
# LOC1   |      |     |
# LOC2   |      |     |
# OFFSET |      |     |
# GROUPT |      |     |
# DEFORM |      |     |
#############################################

concatdf = pd.DataFrame()


#############################################
# TIME ELAPSED | SIM1   | SIM2 | ... | SIM N
# 0            | DATA   |      |     |
# ...          | DATA   |      |     |
# END          | DATA   |      |     |
#############################################

timelimit = 200
rows = 500
cols = 2000

visual = False

j = 0
loops = 20
for i in range(loops):
    if i != 0:
        concatdf = pd.read_csv("MoltenSaltDataframe.csv", index_col = [0])
        concatparameters = pd.read_csv("MoltenSaltParameters.csv", index_col = [0])

    print('SIMULATION ', str(i+1))

    dfparameters = pd.DataFrame()
    dfdata = pd.DataFrame()


    radius = InitialConditions.InitialRadius(0)
    locations = InitialConditions.InitialLocations(0, 0, radius)
    location1 = locations[0]
    location2 = locations[1]
    
    starting_coords = InitialConditions.InitialCoords(radius)
    starting_x = starting_coords[0]
    starting_y = starting_coords[1]

    #if visual == True:
        #results = DatasetConstruction.DataConstruction(location1, location2, radius, starting_x, starting_y, timelimit, rows, cols)
    #else:
    start = timer()
    results = DatasetConstructionNonVisual.DataConstructionNonVisual(location1, location2, radius, starting_x, starting_y, timelimit, rows, cols)
    dt = timer() - start
    print("Computed in %f s" % dt)
    
    calculatedtime = results[1]
    deformation = results[2]
    data = results[0][0]
    
    if i == 0:
        timelist = np.arange(0, len(data), 1)
        dfdata['Time Elapsed'] = timelist
        concatdf['Time Elapsed'] = dfdata['Time Elapsed']

    name = "Cross Correlation Sim " + str(i+1)
    dfdata[name] = data
    dfparameters[name] = np.array([radius, starting_x, starting_y, location1, location2, location2 - location1, calculatedtime, deformation])

    concatdf[name] = dfdata[name]
    concatparameters[name] = dfparameters[name]

    concatdf.to_csv("MoltenSaltDataframe.csv")
    concatparameters.to_csv("MoltenSaltParameters.csv")

