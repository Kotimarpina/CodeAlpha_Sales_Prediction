import pandas as pd
data = pd.read_csv("Advertising_Original.csv")
data = data.drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates()
data.to_csv("Advertising_Cleaned.csv", index=False)
print("Preprocessing completed:", data.shape)
print(data.isnull().sum())
