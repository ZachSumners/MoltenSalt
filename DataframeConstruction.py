import DatasetConstruction
import pandas as pd
import time

data = []
location1s = []
location2s = []
offsets = []

for i in range(5):
    location1 = 50*i + 150
    location2 = 50*i + 250
    results = DatasetConstruction.DataConstruction(location1, location2)
    #
    time.sleep(20)
    print(results)
    data.append(results[0])
    location1s.append(results[1])
    location2s.append(results[2])
    offsets.append(results[3])

df = pd.DataFrame(
    {
        "Data": data,
        "Location 1": location1s,
        "Location 2": location2s,
        "Offset": offsets
    }
)

print(df["Offset"])
