#Critical repeated functions for simulation.
import InitialConditions
import math
import numpy as np
from numba import jit

#Calculates how much deformation occured between two locations.
def deformation_calc(structure):
    lowest = min(structure)
    highest = max(structure) 

    return (highest - lowest)

#Simulation initialization. Generate continuous turbulent structure. Eddy shape currently limited to a circle (technically a hyperbola if you count the value of the grid point).
def spawn_structure(starting_x, starting_y, rows, cols, radius, x, y):
        #Track which pixels are apart of the eddy
        structure = []
        eddy = np.zeros((rows+1,cols+1))

        if starting_x < radius or starting_x > rows-radius or starting_y < radius or starting_y > rows-radius:
            print("Invalid eddy coordinates.")
            return [[0]]

        for x_val in x:
            for y_val in y:
                if ((x_val-(starting_x))**2+(y_val-(starting_y))**2 <= radius**2):
                    eddy[int(x_val)][int(y_val)] = 2*(1-(x_val-starting_x)**2/radius**2 - (y_val-starting_y)**2/radius**2)
                    #Add eddy pixels to list for tracking.
                    structure.append([int(x_val), int(y_val)])
        
        return (eddy, structure)

#Function to find the nearest discrete step to the transducer location (for target values greater than input)
def find_nearest_above(array, target):
    array = np.asarray(array)
    distance_from_target = array - target
    return np.where(distance_from_target > 0, distance_from_target, np.inf).argmin()

#Function to find the nearest discrete step to the transducer location (for target values less than input)   
def find_nearest_below(array, target):
    array = np.asarray(array)
    distance_from_target = array - target
    return np.where(distance_from_target < 0, distance_from_target, -np.inf).argmax()

#Calculates time for structure to cross between transducers.
def time_cross(closest, location, rows, starting_x):
    difference = location - closest
    
    speed = math.floor(VelocityFunction(starting_x, rows))
    crossed = difference/speed
    return crossed

#Calculates the group velocity of the turbulent structure taking into account the discrete steps of the turbulent structure on the grid
def group_velocity_value(means, location1, location2, rows, starting_x):
    middleloc1 = find_nearest_above(means, location1)
    middleloc2 = find_nearest_below(means, location2)
    
    fullsteps = middleloc2 - middleloc1

    loc1cross = abs(time_cross(means[middleloc1], location1, rows, starting_x))
    loc2cross = time_cross(means[middleloc2], location2, rows, starting_x)
    

    return (loc1cross + fullsteps + loc2cross)

#Calculates how much deformation has occured between the transducers taking into account the discrete steps of the turbulent structure on the grid.
def deformation_value(means, deformations, location1, location2):
    index1 = find_nearest_above(means, location1)
    index2 = find_nearest_below(means, location2)

    def1 = deformations[index1]
    def2 = deformations[index2]

    return (def2 - def1)

#SIMULATION VELOCITY FUNCTION.
@jit(nopython=True)
def VelocityFunction(i, size):
    return 2*10*(1-((i-size/2)/(size/2))**2)

#Probably the most important function. Moves all grid points according to velocity function for each frame. GPU accelerated.
@jit(nopython=True)
def flow(z, rows, noise, multiplier):
    for i in range(len(z)):
        shift = multiplier*math.floor(VelocityFunction(i, rows))
        #shift = multiplier*math.floor(-(i-500/2)**2/5000 + 12.5)
        if shift == 0 and i != 0 and i != 499:
            shift = 1
        for j in range(shift):
            if noise == True:
                z[i] = np.append([2*np.random.random()-1], z[i][:-1]) #Between -1 and 1
                #z[i] = np.append([np.random.random()], z[i][:-1]) #Between 0 and 1
            else:
                z[i] = np.append([0], z[i][:-1])
    
    return z
