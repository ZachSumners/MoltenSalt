#MANUAL DEFINITIONS
#Set to 0 if you do NOT want to manually set them.
import numpy as np
import math

def InitialLocations(location1, location2):
    if location1 == 0:
        location1 = math.floor(np.random.normal(600, 200))
        if location1 <= 0:
            location1 = 1
    if location2 == 0:
        location2 = math.floor(np.random.normal(1100, 200))
        if location2 <= location1:
            location2 = location1 + 1
        if location2 >= 2000:
            location2 = 1999
    return [location1, location2]

def InitialRadius(radius):
    if radius == 0:
        radius = math.floor(np.random.normal(100, 10))
    return radius

def InitialCoords(radius):
    x = math.floor(np.random.normal(250, 50))
    y = radius + 1
    if radius >= x:
        x = radius + 1
    if 500 - radius <= x:
        x = 500 - radius - 1
    return [x, y]

def VelocityFunction(i, size):
    #return -(i-size/2)**2/5000 + 12.5'
    #if i <= 250:
    return 15
    #else:
        #return -i/20 + 25

#def EddyShape()