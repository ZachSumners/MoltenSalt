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

    if starting_x < radius or starting_x > rows-radius or starting_y < radius or starting_y > rows-radius:
        print("Invalid eddy coordinates.")
        return [[0]]

    for x_val in x:
        for y_val in y:
            if ((x_val-(rows/2))**2+(y_val-radius)**2 <= radius**2):
                z[int(x_val)][int(y_val)] = 1#(radius-((x_val-(starting_x))**2/radius + (y_val-starting_y)**2/radius))/radius

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
    crosscorrelation.setXRange(0, length_time, padding=0)
    crosscorrelation.setYRange(-100, 200, padding=0)
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
                #shift = 20
                
                #if i < (rows/2):
                #    shift = math.floor(((i + 1)/5))
                #else:
                #    shift = math.floor(((-i+rows)/5))
                shift = math.floor(InitialConditions.VelocityFunction(i, rows))
                if shift == 0:
                    shift = 1
                for j in range(shift):
                    z[i] = np.append([2*np.random.random()-1], z[i][:-1]) #Between -1 and 1
                    #z[i] = np.append([np.random.random()], z[i][:-1]) #Between 0 and 1

                ColumnValue = LineIntegralSum(location1, z[i], ColumnValue)
                ColumnValue2 = LineIntegralSum(location2, z[i], ColumnValue2)    
            ColumnArray = np.array([ColumnValue])
            NewLineSum = np.append(LineSum, ColumnArray)
            ColumnArray2 = np.array([ColumnValue2])
            NewLineSum2 = np.append(LineSum2, ColumnArray2)
            
            wGrid.setImage(z)

            LineIntegralPlot.setData(NewLineSum)
            LineIntegralPlot2.setData(NewLineSum2)
            print('{:.0f} FPS'.format(1 / (time.time() - stime)))

        else:     
            CorrelationList = np.correlate(NewLineSum2, NewLineSum, mode='full')
            middle = rows/2

            crosscorrelationPlot.setData(CorrelationList)
            crosscorrelationData.append(CorrelationList)
            
            end = True
            w.close()
            

    #Sum the grid cross section as the signal data point.
    def LineIntegralSum(columnNumber, row, ColumnValue):
        ColumnValue += row[columnNumber]
        return ColumnValue
    
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10)

    #if __name__ == '__main__':
    pg.exec()
  
    return crosscorrelationData

#Module supports
if __name__ == "__main__":
   DataConstruction(location1, location2)