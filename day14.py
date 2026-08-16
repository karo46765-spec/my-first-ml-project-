from day11 import prediction
from day10 import new_applicant
from day11 import X_train
import numpy as nm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train = nm.array([[540, 30000],[590, 45000],[630,50000],[710,85000],[780,120000]])

y_train = nm.array([1,1,1,0,0])

model = LogisticRegression()

Scaler = StandardScaler()

X_scale = Scaler.fit_transform(X_train)

print(f"Scaled Data : {X_scale}")

model.fit(X_scale,y_train)

new_applicant = nm.array([[680, 70000],[520,89000],[120,150000]])

prediction = model.predict(new_applicant)

print(f"Prediction : {prediction}")