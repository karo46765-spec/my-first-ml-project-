import numpy as nm
from sklearn.preprocessing import StandardScaler

X_raw = nm.array([[22,30000],[45,85000],[60,120000]])

print("---Original Raw feature___")
print(X_raw)

scaler =  StandardScaler()

Scaleddata = scaler.fit_transform(X_raw)


print("\n---Standard features (Mean = 0, std=1)---")
print(Scaleddata)

print(f"\nCalculated Mean per column :{scaler.mean_}")
