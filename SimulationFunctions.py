#Critical repeated functions for simulation.
import InitialConditions
import math
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

#Calculates how much deformation occured between two locations.
def deformation_calc(structure):
    lowest = min(structure)
    highest = max(structure) 

    return (highest - lowest)

#Simulation initialization. Generate continuous turbulent structure. Eddy shape currently limited to a circle (technically a hyperbola if you count the value of the grid point).
def spawn_structure(starting_y, starting_x, rows, cols, radius, x, y):
        #Initialize eddy
        eddy = np.zeros((rows,cols))

        #Check to see if it's out of bounds
        if starting_x < radius or starting_x > rows-radius or starting_y < radius or starting_y > rows-radius:
            print("Invalid eddy coordinates.")
            return [[0]]

        #Generate eddy with a mask and parametric equation of circle brightness (darker on edges)
        Y, X = np.ogrid[:rows, :cols]
        mask = ((X-starting_x)**2+(Y-starting_y)**2) <= radius**2
        eddy = 2*(1-(X-starting_x)**2/radius**2 - (Y-starting_y)**2/radius**2) * mask

        return eddy, mask

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
    
    speed = math.floor(VelocityFunction(rows, starting_x))
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
#@jit(nopython=True)
def VelocityFunction(size, x=None):
    if x == None:
        x = np.arange(0, size)
    return 2*10*(1-((x-size/2)/(size/2))**2)

#Probably the most important function. Moves all grid points according to velocity function for each frame. GPU accelerated.
#@jit(nopython=True)
def flow(z, rows, noise, multiplier):
    #Calculate the shift in each column of the array because flow not uniform
    shift = multiplier*np.round(VelocityFunction(rows))
    shift = np.where(shift == 0, 1, shift)
    shift = np.asarray(shift, dtype=np.int64)

    #Get array shape
    rows, cols = z.shape
    cols_arr = np.arange(cols, dtype=np.int64)

    #Move columns by unique shift number by numpy broadcasting
    shifted_cols = cols_arr[None, :] - shift[:, None] 
    mask = (shifted_cols < 0) | (shifted_cols >= cols) 
    shifted_cols_clipped = np.clip(shifted_cols, 0, cols - 1)

    #Fill each column with however many points went out of bounds (so things are moving in one direction at different rates)
    z_shifted = z[np.arange(rows)[:, None], shifted_cols_clipped]
    
    #Either fill with noise or 0s.
    if noise == True:
        new = 2*np.random.random(z.shape)-1
        z_shifted[mask] = new[mask]
    else:
        z_shifted[mask] = 0
    
    z = z_shifted

    return z
