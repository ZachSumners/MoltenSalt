#Relevant library and module imports.
#import DatasetConstruction
import InitialConditions
import DatasetConstructionNonVisual
import DatasetConstruction
import pandas as pd
import numpy as np
import math
from timeit import default_timer as timer


#Create two dataframes. One to store the initial conditions of each simulation run (concatparameters) and one to store the actual cross correlation data for that run (concatdf)
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


#Controls maximum number of frames a simulation can have.
timelimit = 200
#Controls size of simulation grid for all iterations.
rows = 500
cols = 2000

#Controls whether you want the simulation grid to show up and run with pyqtgraph.
visual = False

#How many simulation iterations to run
loops = 3

#Length of existing dataframe to know what simulation number to start at.
namedf = 'MoltenSaltDataframe.csv'
nameparams = 'MoltenSaltParameters.csv'
try:     
    concatdf = pd.read_csv(namedf, index_col = [0])
    concatparameters = pd.read_csv(nameparams, index_col = [0])
    lengthdataframe = len(concatdf.columns)-1
except:
    lengthdataframe = 0

#Run the simulation loop number of times.
for i in range(lengthdataframe, lengthdataframe+loops, 1):
    #If there are existing dataframes, append new simulation runs to them. If not, create new ones.
    try:     
        concatdf = pd.read_csv("MoltenSaltDataframe.csv", index_col = [0])
        concatparameters = pd.read_csv("MoltenSaltParameters.csv", index_col = [0])
    except:
        if i != 0:
            concatdf = pd.read_csv("MoltenSaltDataframe.csv", index_col = [0])
            concatparameters = pd.read_csv("MoltenSaltParameters.csv", index_col = [0])

    #Simulation number tracker
    print('SIMULATION ', str(i+1))

    #Initialize dataframe to store simulation iteration data.
    dfparameters = pd.DataFrame()
    dfdata = pd.DataFrame()

    #Generate the initial conditions of the simulation.
    #**Set the function parameter to 0 if you want it to be random. Set to any int if you want to specify it.**
    radius = InitialConditions.InitialRadius(0)
    locations = InitialConditions.InitialLocations(0, 0, radius)
    location1 = locations[0]
    location2 = locations[1]
    
    starting_coords = InitialConditions.InitialCoords(radius)
    starting_x = starting_coords[0]
    starting_y = starting_coords[1]

    #Run the simulation module, tracking how long it took. GPU acceleration speeds it up by about x3.

    if visual == True:
        results = DatasetConstruction.DataConstruction(location1, location2, radius, starting_x, starting_y, timelimit, rows, cols)
    else:
        start = timer()
        results = DatasetConstructionNonVisual.DataConstructionNonVisual(location1, location2, radius, starting_x, starting_y, timelimit, rows, cols)
        dt = timer() - start
        print("Computed in %f s" % dt)
    
    #Values returned from simulation stored in variables.
    calculatedtime = results[1]
    deformation = results[2]
    data = results[0][0]
    
    #Create the "Time Elapsed" pandas column.
    if i == 0:
        timelist = np.arange(0, len(data), 1)
        dfdata['Time Elapsed'] = timelist
        concatdf['Time Elapsed'] = dfdata['Time Elapsed']

    #Store simulation data in a pandas dataframe.
    name = "Cross Correlation Sim " + str(i+1)
    dfdata[name] = data
    dfparameters[name] = np.array([radius, starting_x, starting_y, location1, location2, location2 - location1, calculatedtime, deformation])

    concatdf[name] = dfdata[name]
    concatparameters[name] = dfparameters[name]

    #Save new appended dataframe.
    concatdf.to_csv("MoltenSaltDataframe.csv")
    concatparameters.to_csv("MoltenSaltParameters.csv")

