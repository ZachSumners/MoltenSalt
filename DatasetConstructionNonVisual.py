import numpy as np
import scipy.signal
import time
import SimulationFunctions
from numba import jit
import matplotlib.pyplot as plt

#Constructs the molten salt datasets but without plotting.

def DataConstructionNonVisual(location1, location2, radius, starting_x, starting_y, length_time, rows, cols):
    global SumPlot_y, SumPlot_y2, end, NewLineSum, NewLineSum2, counter

    #Initialize the random pipe cross section (grid) and the single eddy at the beginning.
    x = np.linspace(0, rows, rows+1)
    y = np.linspace(0, cols, cols+1)
    noiseOverlay = 2*np.random.random((rows+1,cols+1)) -1 #Between -1 and 1
    
    structure_func = SimulationFunctions.spawn_structure(starting_x, starting_y, rows, cols, radius, x, y)
    structure1overlay = structure_func[0]

    #Controls how long the simulation runs (in frames)
    counter = 0

    #Plotting initializations
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
        
        counter += 1
        ColumnValue = 0
        ColumnValue2 = 0
        LineSum = NewLineSum
        LineSum2 = NewLineSum2

        #Invalid initialization catch.
        if counter < 0:
            return
        
        #If the simulation has not finished, calculate new line integrals (ultrasonic measurement) for each measurement location and move the grid to simulate flow.
        if counter < length_time:           
            #Move the seperate parts of the grid based on velocity profile
            noiseOverlay = SimulationFunctions.flow(noiseOverlay, rows, True, 1)
            structure1overlay = SimulationFunctions.flow(np.asarray(structure1overlay), rows, False, 1)
            #z is the combination of grid components.
            z = noiseOverlay + structure1overlay

            #Calculate line integrals for each location and store in list.
            ColumnValue = np.sum(z, axis=0)[location1]
            ColumnValue2 = np.sum(z, axis=0)[location2]
            NewLineSum = np.append(LineSum, np.array([ColumnValue]))
            NewLineSum2 = np.append(LineSum2, np.array([ColumnValue2]))

            #Track how the overall structure envelope is moving for group velocity calculation.
            structurecoords = [coords[1] for coords in np.argwhere(structure1overlay > 0)]
            totalcount = sum(structurecoords)
            numStructureRows = len(structurecoords)

            #Calculate how much the structure has deformed (compared with how it began) once it has passed through the second measurement location.
            if numStructureRows != 0:
                structure1GroupVel = totalcount/numStructureRows
                means.append(structure1GroupVel)
                deformations.append(SimulationFunctions.deformation_calc(structurecoords))

            return [noiseOverlay, structure1overlay, []]

        #When the simulation run is over and the two ultrasonic signals have been gathered, correlate them.
        else: 
            
            plt.plot(np.arange(0, len(NewLineSum2), 1), NewLineSum2)
            plt.xlabel('Time')
            plt.ylabel('Strength')
            plt.title('Ultrasonic Measurement - Location 2')
            plt.grid()
            plt.show()

            plt.plot(np.arange(0, len(NewLineSum), 1), NewLineSum)
            plt.xlabel('Time')
            plt.ylabel('Strength')
            plt.title('Ultrasonic Measurement - Location 1')
            plt.grid()
            plt.show()

            CorrelationList = scipy.signal.correlate(NewLineSum2, NewLineSum, mode='full')
            crosscorrelationData.append(CorrelationList)
            return [noiseOverlay, structure1overlay, crosscorrelationData]
    
    #Run the simulation for length_time frames, updating each grid component each frame.
    for counter in range(length_time):
        overlays = update(noiseOverlay, structure1overlay)
        noiseOverlay = overlays[0]
        structure1overlay = overlays[1]
        crosscorrelationData = overlays[2]
    
    #Calculate group velocity and overall deformation of the turbulent structure.
    deformation = SimulationFunctions.deformation_value(means, deformations, location1, location2)
    groupvel1 = SimulationFunctions.group_velocity_value(means, location1, location2, rows, starting_x)

    return [crosscorrelationData, groupvel1, deformation]
    

#Module supports
if __name__ == "__main__":
   DataConstruction(location1, location2)