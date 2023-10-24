#LEGACY. Rest of the simulation was built on this.

"""
This example demonstrates the use of GLSurfacePlotItem.
"""

import numpy as np
import math
import random
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import time
from pyqtgraph.Qt import QtWidgets
import matplotlib.pyplot as plt
from scipy import signal

## Create a GL View widget to display data
app = pg.mkQApp("GLSurfacePlot Example")
w = gl.GLViewWidget()
#w.show()
w.setWindowTitle('pyqtgraph example: GLSurfacePlot')
w.setCameraPosition(distance=100)

rows = 300
cols = 1000
x = np.linspace(0, rows, rows)
y = np.linspace(0, cols, cols)
z = 2*np.random.random((rows,cols)) -1

p2 = gl.GLSurfacePlotItem(x=x, y=y, z=z, shader='normalColor', smooth=True)
p2.translate(0,0,0)
w.addItem(p2)

index = 0
NewLineSum = np.array([])
NewLineSum2 = np.array([])

win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

pL1 = win.addPlot(title="Line Integral Location 1")
pL1.setXRange(0, 250, padding=0)
LineIntegralPlot = pL1.plot(pen='y')

win.nextRow()

pL2 = win.addPlot(title="Line Integral Location 2")
pL2.setXRange(0, 250, padding=0)

LineIntegralPlot2 = pL2.plot(pen='y')

win2 = QtWidgets.QMainWindow()
win2.resize(800,800)
imv = pg.ImageView()
win2.setCentralWidget(imv)
win2.show()
win2.setWindowTitle('pyqtgraph example: ImageView')
cmap = pg.colormap.get('CET-L1')
imv.setColorMap(cmap)


#win3 = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples2")
#win3.resize(1000,600)
#win3.setWindowTitle('pyqtgraph example: Plotting')

#pD1 = win3.addPlot(title="Line Integral Distance")
#pD1.setXRange(0, rows, padding=0)
#LineIntegralPositionPlot = pD1.plot(pen='y')

#win3.nextRow()

#pD2 = win3.addPlot(title="Line Integral Distance")
#pD2.setXRange(0, rows, padding=0)
#LineIntegralPositionPlot2 = pD2.plot(pen='y')


win4 = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples2")
win4.resize(1000,600)
win4.setWindowTitle('pyqtgraph example: Plotting') 
crosscorrelation = win4.addPlot(title="Cross Correlation")
crosscorrelation.setXRange(0, 50, padding=0)
crosscorrelationPlot = crosscorrelation.plot(pen='y')

first = 200
second = 500
SumPlot_y = []
SumPlot_y2 = []
end = False

CorrelationList = []

counter = 0
def update():
    global counter, NewLineSum, NewLineSum2, p2, z, index, tau, end
    stime = time.time()
    index -= 1
    counter += 1
    ColumnValue = 0
    ColumnValue2 = 0
    ColumnValuePlot = 0
    ColumnValuePlot2 = 0
    LineSum = NewLineSum
    LineSum2 = NewLineSum2
    if end == True:
        return
    if counter < 200:
        for i in range(len(z)):
            shift = 5
            #if i < (rows/2):
                #shift = math.floor(((i + 1)/5))
            #else:
                #shift = math.floor(((-i+rows)/5))
            #if shift == 0:
                #shift = 1
            for j in range(shift):
                z[i] = np.append([2*np.random.random()-1], z[i][:-1])
            ColumnValue = LineIntegralSum(200, z[i], ColumnValue)
            ColumnValue2 = LineIntegralSum(500, z[i], ColumnValue2)    
        ColumnArray = np.array([ColumnValue])
        NewLineSum = np.append(LineSum, ColumnArray)
        ColumnArray2 = np.array([ColumnValue2])
        NewLineSum2 = np.append(LineSum2, ColumnArray2)

        #if counter == first:
        #    for j in range(len(z)):
        #        ColumnValuePlot = 0
        #        ColumnValuePlot = LineIntegralSum(j, z[j], ColumnValuePlot)
        #        SumPlot_y.append(ColumnValuePlot)
        #    LineIntegralPositionPlot.setData(SumPlot_y)
        #if counter == second:
        ##    for j in range(len(z)):
         #       ColumnValuePlot2 = 0
         #       ColumnValuePlot2 = LineIntegralSum(j, z[j], ColumnValuePlot2)
         #       SumPlot_y2.append(ColumnValuePlot2)
         #   LineIntegralPositionPlot2.setData(SumPlot_y2)
        
        #p2.setData(z=z)
        imv.setImage(z)
        

        LineIntegralPlot.setData(NewLineSum)
        LineIntegralPlot2.setData(NewLineSum2)
        print('{:.0f} FPS'.format(1 / (time.time() - stime)))

    else:
        #corr = signal.correlate(NewLineSum, NewLineSum2)/len(NewLineSum)
        
        for tau in range(len(NewLineSum)-1):
            CorrelationSum = 0
            for t in range(len(NewLineSum)-tau):
                CorrelationSum += NewLineSum[t]*NewLineSum2[t+tau]
            CorrelationSum = CorrelationSum/(len(NewLineSum)-tau)
        #Correlation = np.correlate(NewLineSum, NewLineSum2)
        #print(Correlation)
            CorrelationList.append(CorrelationSum)
        crosscorrelationPlot.setData(CorrelationList)   
        end = True

def LineIntegralSum(columnNumber, row, ColumnValue):
    ColumnValue += row[columnNumber]
    return ColumnValue
 
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)


if __name__ == '__main__':
    pg.exec()