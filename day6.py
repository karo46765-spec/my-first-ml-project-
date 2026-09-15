import pandas as pd

durl = (
    "https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv"
)

df = pd.read(durl)

print(df.shape())
print(df.head())
pyt
