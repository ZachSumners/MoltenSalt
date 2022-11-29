from turtle import st
import numpy as np
import scipy.signal
import math
import time
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import InitialConditions
import SimulationFunctions

import sys

def DataConstructionNonVisual(location1, location2, radius, starting_x, starting_y, length_time, rows, cols):
    global SumPlot_y, SumPlot_y2, end, NewLineSum, NewLineSum2, p1, counter, z, app

    #Initialize the random pipe cross section (grid) and the single eddy at the beginning.
    x = np.linspace(0, rows, rows+1)
    y = np.linspace(0, cols, cols+1)
    noiseOverlay = 2*np.random.random((rows+1,cols+1)) -1 #Between -1 and 1
    #z = np.random.random((rows+1, cols+1)) #Between 0 and 1
    
    structure_func = SimulationFunctions.spawn_structure(starting_x, starting_y, rows, cols, radius, x, y)
    structure1overlay = structure_func[0]
    structure = structure_func[1]

    structure2overlayInitial = np.zeros((rows+1,cols+1))

    #Controls how long the simulation runs (in frames)
    counter = 0

    #Plotting initializations
    CorrelationList = []
    SumPlot_y = []
    SumPlot_y2 = []
    end = False
    NewLineSum = np.array([])
    NewLineSum2 = np.array([])

    #Pd dataframe prestorage
    crosscorrelationData = []

    #Group velocity lists
    means = []
    means2 = []
    loc1track = []
    loc2track = []

    def update(noiseOverlay, structure1overlay):
        global counter, NewLineSum, NewLineSum2, end
        stime = time.time()
        
        counter += 1
        ColumnValue = 0
        ColumnValue2 = 0
        LineSum = NewLineSum
        LineSum2 = NewLineSum2

        if counter < 0:
            return
        
        if counter < length_time:

            #if counter == 50:
            #    structure_func2 = SimulationFunctions.spawn_structure(starting_x, starting_y, rows, cols, radius, x, y)
            #    structure2overlay = structure_func2[0]
            #    global structure2
            #    structure2 = structure_func2[1]
            
            #if counter >= 50:
            #    noiseOverlay = SimulationFunctions.flow(noiseOverlay, rows, True, 1)
            #    structure1overlay = SimulationFunctions.flow(np.asarray(structure1overlay), rows, False, 1)
            #    structure2overlay = SimulationFunctions.flow(np.asarray(structure2overlay), rows, False, 2)
            #    z = noiseOverlay + structure1overlay + structure2overlay
            #else:
            noiseOverlay = SimulationFunctions.flow(noiseOverlay, rows, True, 1)
            structure1overlay = SimulationFunctions.flow(np.asarray(structure1overlay), rows, False, 1)
            z = noiseOverlay + structure1overlay

            ColumnValue = np.sum(z, axis=0)[location1]
            ColumnValue2 = np.sum(z, axis=0)[location2]
            NewLineSum = np.append(LineSum, np.array([ColumnValue]))
            NewLineSum2 = np.append(LineSum2, np.array([ColumnValue2]))

            structure1GroupVel = SimulationFunctions.group_velocity_calc(structure, location1, location2, rows, means, loc1track, loc2track)
            means.append(structure1GroupVel[0])
            #loc1track.append(structure1GroupVel[1])
            #loc2track.append(structure1GroupVel[2])

            #if counter > 50:
            #    structure2GroupVel = SimulationFunctions.group_velocity_calc(structure2, location1, location2, rows, means, loc1track, loc2track)
            #    means2.append(structure2GroupVel[0])
                #loc1track.append(structure1GroupVel[1])
                #loc2track.append(structure1GroupVel[2])
            return [noiseOverlay, structure1overlay, []]


        else:   
            CorrelationList = scipy.signal.correlate(NewLineSum2, NewLineSum, mode='full')
            crosscorrelationData.append(CorrelationList)
            return [noiseOverlay, structure1overlay, crosscorrelationData]
    
    for counter in range(length_time):
        overlays = update(noiseOverlay, structure1overlay)
        noiseOverlay = overlays[0]
        structure1overlay = overlays[1]
        crosscorrelationData = overlays[2]
    

    groupvel1 = SimulationFunctions.group_velocity_value(means, location1, location2, rows, starting_x)
    #groupvel2 = SimulationFunctions.group_velocity_value(means2, location1, location2, rows, starting_x)

    #print(groupvel1)#, groupvel2)

    #loc1track = np.asarray(loc1track)
    #loc2track = np.asarray(loc2track)
    #maxloc1 = np.argmax(loc1track)
    #maxloc2 = np.argmax(loc2track)
    #print(maxloc2 - maxloc1)

    return [crosscorrelationData, groupvel1]
    

#Module supports
if __name__ == "__main__":
   DataConstruction(location1, location2)