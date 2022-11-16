from turtle import st
import wave
import numpy as np
import math
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import time
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import InitialConditions
import scipy.signal

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
    z = 2*np.random.random((rows+1,cols+1)) -1 #Between -1 and 1
    #z = np.random.random((rows+1, cols+1)) #Between 0 and 1
    #radius = 50

    #Track which pixels are apart of the eddy
    structure = []

    if starting_x < radius or starting_x > rows-radius or starting_y < radius or starting_y > rows-radius:
        print("Invalid eddy coordinates.")
        return [[0]]

    for x_val in x:
        for y_val in y:
            if ((x_val-(starting_x))**2+(y_val-(starting_y))**2 <= radius**2):
                z[int(x_val)][int(y_val)] = 1-(x_val-starting_x)**2/radius**2 - (y_val-starting_y)**2/radius**2
                #Add eddy pixels to list for tracking.
                structure.append([int(x_val), int(y_val)])
    
    print(z.shape)
    #Plot the grid with pyqtgraph.
    d1.hideTitleBar()
    wGrid = pg.ImageView()
    p1 = wGrid.setImage(z)
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

    #** LOCATION OF SENSORS IN PX ALONG PIPE CROSS SECTION**
    #location1 = 200
    #location2 = 300

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
    means = []
    loc1track = []
    loc2track = []

    def update():
        global counter, NewLineSum, NewLineSum2, z, end
        stime = time.time()
        
        counter += 1
        ColumnValue = 0
        ColumnValue2 = 0
        LineSum = NewLineSum
        LineSum2 = NewLineSum2
        if end == True:
            return
        if counter < 0:
            return
        if counter < length_time:
            for i in range(len(z)):
                shift = math.floor(InitialConditions.VelocityFunction(i, rows))
                if shift == 0:
                    shift = 1
                for j in range(shift):
                    z[i] = np.append([2*np.random.random()-1], z[i][:-1]) #Between -1 and 1
                    #z[i] = np.append([np.random.random()], z[i][:-1]) #Between 0 and 1
  
            ColumnValue = np.sum(z, axis=0)[location1]
            ColumnValue2 = np.sum(z, axis=0)[location2]
            NewLineSum = np.append(LineSum, np.array([ColumnValue]))
            NewLineSum2 = np.append(LineSum2, np.array([ColumnValue2]))

            loc1count = 0
            loc2count = 0
            for c in range(len(structure)):
                if structure[c][1] == location1:
                    loc1count += 1
                if structure[c][1] == location2:
                    loc2count += 1
                
                shift_border = math.floor(InitialConditions.VelocityFunction(structure[c][0], rows))
                structure[c][1] += shift_border

            loc1track.append(loc1count)
            loc2track.append(loc2count)
            
            mean = sum(elt[1] for elt in structure)/len(structure)        
            means.append(mean)

            
            wGrid.setImage(z)

            LineIntegralPlot.setData(NewLineSum)
            LineIntegralPlot2.setData(NewLineSum2)
            #print('{:.0f} FPS'.format(1 / (time.time() - stime)))

        else:     
            CorrelationList = scipy.signal.correlate(NewLineSum2, NewLineSum, mode='full')
            

            crosscorrelationPlot.setData(CorrelationList/max(CorrelationList))
            crosscorrelationData.append(CorrelationList)
            
            end = True
            w.close()
    
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(2)

    #if __name__ == '__main__':
    pg.exec()

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def time_cross(closest, location, rows, starting_x):
        difference = location - closest
        
        speed = math.floor(InitialConditions.VelocityFunction(starting_x, rows))
        crossed = difference/speed
        return crossed

    middleloc1 = find_nearest(means, location1)
    middleloc2 = find_nearest(means, location2)

    loc1cross = time_cross(middleloc1, location1, rows, starting_x)
    timeone = means.index(middleloc1) + loc1cross
    loc2cross = time_cross(middleloc2, location2, rows, starting_x)
    timetwo = means.index(middleloc2) + loc2cross
    #print(timetwo - timeone)

    #loc1track = np.asarray(loc1track)
    #loc2track = np.asarray(loc2track)
    #maxloc1 = np.argmax(loc1track)
    #maxloc2 = np.argmax(loc2track)
    #print(maxloc2 - maxloc1)

    return [crosscorrelationData, timetwo-timeone]

#Module supports
if __name__ == "__main__":
   DataConstruction(location1, location2)