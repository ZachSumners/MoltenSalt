import pandas as pd

myData = pd.DataFrame(
    {
        "Name": ["A", "B", "C"],
        "Age": [34, 65, 14],
        "Sex": ["Male", "Male", "Female"],
        "Favorite Numbers": [[1,2,3], [4,5,6], [12, 21, 55]]
    }
)

myData.to_csv("TestPandasCSV.csv")