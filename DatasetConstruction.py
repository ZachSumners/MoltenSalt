import numpy as np
import scipy.signal
import time
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import SimulationFunctions
import matplotlib.pyplot as plt

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
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    noiseOverlay = 2*np.random.random((rows,cols)) -1 #Between -1 and 1
    #z = np.random.random((rows, cols)) #Between 0 and 1
    
    structure1overlay, structure = SimulationFunctions.spawn_structure(starting_x, starting_y, rows, cols, radius, x, y)

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
    deformations = []

    def update(noiseOverlay, structure1overlay):
        global counter, NewLineSum, NewLineSum2, end
        stime = time.time()
 
        ColumnValue = 0
        ColumnValue2 = 0
        LineSum = NewLineSum
        LineSum2 = NewLineSum2  

        #time.sleep(2)    

        if counter < length_time and end == False:
            counter += 1
            #Simulate the grid points flowing as defined by the velocity function.
            noiseOverlay_new = SimulationFunctions.flow(noiseOverlay, rows, True, 1)
            np.copyto(noiseOverlay, noiseOverlay_new) 

            new_structure1overlay = SimulationFunctions.flow(np.asarray(structure1overlay), rows, False, 1)
            np.copyto(structure1overlay, new_structure1overlay) 
            #Add the two components
            z = noiseOverlay + structure1overlay

            #Store line integral transducer calculations for each frame.
            ColumnValue = np.sum(z, axis=0)[location1]
            ColumnValue2 = np.sum(z, axis=0)[location2]
            NewLineSum = np.append(LineSum, np.array([ColumnValue]))
            NewLineSum2 = np.append(LineSum2, np.array([ColumnValue2]))

            #Track where all the grid points of the turbulent structure are.
            structurecoords = [coords[1] for coords in np.argwhere(structure1overlay > 0)]
            totalcount = sum(structurecoords)
            numStructureRows = len(structurecoords)

            if numStructureRows != 0:
                #Group velocity and deformation value tracking.
                structure1GroupVel = totalcount/numStructureRows
                means.append(structure1GroupVel)
                deformations.append(SimulationFunctions.deformation_calc(structurecoords))
                #Update the grid plot.
                wGrid.setImage(structure1overlay)
                #Update the transducer plots
                LineIntegralPlot.setData(NewLineSum)
                LineIntegralPlot2.setData(NewLineSum2)
                #print('{:.0f} FPS'.format(1 / (time.time() - stime)))
            else:
                end = True

        else:   
            #Perform the cross correlation and end the simulation. 
            
            CorrelationList = scipy.signal.correlate(NewLineSum2, NewLineSum, mode='full')
            crosscorrelationPlot.setData(CorrelationList/max(CorrelationList))
            crosscorrelationData.append(CorrelationList)

            crosscorrelationPlot.setData(CorrelationList)
            w.update()

            end = True
            time.sleep(3)
            w.close()
            return
    
    #Function to update the visual grid.
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: update(noiseOverlay, structure1overlay))
    timer.start(1)

    #if __name__ == '__main__':
    pg.exec()
    
    #Calculates total deformation and group velocity of the structure
    deformation = SimulationFunctions.deformation_value(means, deformations, location1, location2)
    groupvel1 = SimulationFunctions.group_velocity_value(means, location1, location2, rows, starting_x)

    return crosscorrelationData, groupvel1, deformation
    

#Module supports
if __name__ == "__main__":
   DataConstruction(location1, location2)
