import InitialConditions
import math
import numpy as np

def group_velocity_calc(structure, location1, location2, rows, means, loc1track, loc2track):
        loc1count = 0
        loc2count = 0
        for c in range(len(structure)):
            if structure[c][1] == location1:
                loc1count += 1
            if structure[c][1] == location2:
                loc2count += 1
            
            shift_border = math.floor(InitialConditions.VelocityFunction(structure[c][0], rows))
            structure[c][1] += shift_border
        
        mean = sum(elt[1] for elt in structure)/len(structure)        

        return (mean, loc1count, loc2count)

def spawn_structure(starting_x, starting_y, rows, cols, radius, z, x, y):
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
        
        return (z, structure)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def time_cross(closest, location, rows, starting_x):
    difference = location - closest
    
    speed = math.floor(InitialConditions.VelocityFunction(starting_x, rows))
    crossed = difference/speed
    return crossed