#MANUAL DEFINITIONS
#Set to 0 if you do NOT want to manually set them.
location1 = 0
location2 = 0
radius = 0

def InitialLocations(location1, location2):
    if location1 == 0:
        location1 = 300
    if location2 == 0:
        location2 = 400
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