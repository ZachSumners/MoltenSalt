#MANUAL DEFINITIONS
#Set to 0 if you do NOT want to manually set them.

def InitialLocations(location1, location2, multiple1, multiple2, starting1, starting2, i):
    if location1 == 0:
        location1 = multiple1 * i + starting1
    if location2 == 0:
        location2 = multiple2 * i + starting2
    return [location1, location2]

def InitialRadius(radius):
    if radius == 0:
        radius = 50
    return radius

#def InitialCoords():
#    x = 150
#    y = 150
#    return [x, y]

#def InitialFlowSpeed():