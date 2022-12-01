from turtle import st
import numpy as np
import numpy.ma as ma
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

def DataConstruction(location1, location2, radius, starting_x, starting_y, length_time, rows, cols):
    global SumPlot_y, SumPlot_y2, end, NewLineSum, NewLineSum2, p1, counter, z, app
    ## Create a GL View widget to display data
    app = pg.mkQApp("Single eddy cross correlation")
    w = QtWidgets.QMainWindow()
    area = DockArea()
    w.setCentralWidget(area)
    w.setWindowTitle('Single eddy cross correlation')
    w.resize(1000, 600)

    #Change plot background and foreground colours.
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    #Create the different "docks" (widgets the different plots live in).
    d1 = Dock("Dock1", size=(300, 500))
    d2 = Dock("Dock 2", size=(400, 300))
    d3 = Dock("Dock 2", size=(400, 300))
    d4 = Dock("Dock 2", size=(400, 300))
    area.addDock(d1, 'left')
    area.addDock(d2, 'right')
    area.addDock(d3, 'bottom', d2)
    area.addDock(d4, 'bottom', d3)

    #Initialize the random pipe cross section (grid) and the single eddy at the beginning.
    x = np.linspace(0, rows, rows+1)
    y = np.linspace(0, cols, cols+1)
    noiseOverlay = 2*np.random.random((rows+1,cols+1)) -1 #Between -1 and 1
    #z = np.random.random((rows+1, cols+1)) #Between 0 and 1
    
    structure_func = SimulationFunctions.spawn_structure(starting_x, starting_y, rows, cols, radius, x, y)
    structure1overlay = structure_func[0]
    structure = structure_func[1]

    structure2overlayInitial = np.zeros((rows+1,cols+1))

    #Plot the grid with pyqtgraph.
    d1.hideTitleBar()
    wGrid = pg.ImageView()
    p1 = wGrid.setImage(noiseOverlay)
    d1.addWidget(wGrid)

    #Plot the first line integral.
    d2.hideTitleBar()
    wPlotLine1 = pg.PlotWidget(title="Ultrasonic Signal, Location 1")
    wPlotLine1.setXRange(0, length_time, padding=0)
    wPlotLine1.setLabel('bottom', 'Time')
    wPlotLine1.setLabel('left', 'Strength')
    LineIntegralPlot = wPlotLine1.plot(pen='k')
    d2.addWidget(wPlotLine1)

    #Plot the second line integral.
    d3.hideTitleBar()
    wPlotLine2 = pg.PlotWidget(title="Ultrasonic Signal, Location 2")
    wPlotLine2.setXRange(0, length_time, padding=0)
    wPlotLine2.setLabel('bottom', 'Time')
    wPlotLine2.setLabel('left', 'Strength')
    LineIntegralPlot2 = wPlotLine2.plot(pen='k')
    d3.addWidget(wPlotLine2)

    #Plot the cross correlation (which gets filled in when the simulation is done.)
    d4.hideTitleBar()
    crosscorrelation = pg.PlotWidget(title="Cross Correlation")
    crosscorrelation.setXRange(0, length_time*2, padding=0)
    crosscorrelation.setYRange(-1, 1, padding=0)
    crosscorrelation.setLabel('bottom', 'Time')
    crosscorrelation.setLabel('left', 'Strength')
    crosscorrelationPlot = crosscorrelation.plot(pen='k')
    d4.addWidget(crosscorrelation)

    #Show the plotting window.
    w.show()

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
    deformations = []

    def update(noiseOverlay, structure1overlay):
        global counter, NewLineSum, NewLineSum2, end
        stime = time.time()
 
        ColumnValue = 0
        ColumnValue2 = 0
        LineSum = NewLineSum
        LineSum2 = NewLineSum2      

        if counter < length_time and end == False:
            counter += 1
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

            structurecoords = [coords[1] for coords in np.argwhere(structure1overlay > 0)]
            totalcount = sum(structurecoords)
            numStructureRows = len(structurecoords)

            if numStructureRows != 0:
                structure1GroupVel = totalcount/numStructureRows
                print(structure1GroupVel)
                means.append(structure1GroupVel)
                deformations.append(SimulationFunctions.deformation_calc(structurecoords))
                    
                #loc1track.append(structure1GroupVel[1])
                #loc2track.append(structure1GroupVel[2])

                #if counter > 50:
                #    structure2GroupVel = SimulationFunctions.group_velocity_calc(structure2, location1, location2, rows, means, loc1track, loc2track)
                #    means2.append(structure2GroupVel[0])
                    #loc1track.append(structure1GroupVel[1])
                    #loc2track.append(structure1GroupVel[2])
                
                wGrid.setImage(structure1overlay)

                LineIntegralPlot.setData(NewLineSum)
                LineIntegralPlot2.setData(NewLineSum2)
                #print('{:.0f} FPS'.format(1 / (time.time() - stime)))
            else:
                end = True

        else:    
            CorrelationList = scipy.signal.correlate(NewLineSum2, NewLineSum, mode='full')
            crosscorrelationPlot.setData(CorrelationList/max(CorrelationList))
            crosscorrelationData.append(CorrelationList)
            
            end = True
            w.close()
            return
    
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: update(noiseOverlay, structure1overlay))
    timer.start(1)


    #if __name__ == '__main__':
    pg.exec()
    
    print(means, deformations)
    deformation = SimulationFunctions.deformation_value(means, deformations, location1, location2)
    groupvel1 = SimulationFunctions.group_velocity_value(means, location1, location2, rows, starting_x)
    
    #groupvel2 = SimulationFunctions.group_velocity_value(means2, location1, location2, rows, starting_x)

    #print(groupvel1)#, groupvel2)

    #loc1track = np.asarray(loc1track)
    #loc2track = np.asarray(loc2track)
    #maxloc1 = np.argmax(loc1track)
    #maxloc2 = np.argmax(loc2track)
    #print(maxloc2 - maxloc1)

    return [crosscorrelationData, groupvel1, deformation]
    

#Module supports
if __name__ == "__main__":
   DataConstruction(location1, location2)