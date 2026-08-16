from day11 import prediction
from day10 import new_applicant
import numpy as nm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb


X = nm.array([[750,0.2,45],[500,0.6,23],[620,0.4,34],[710,0.1,52],[580,0.5,29],[690,0.3,41]])
y= nm.array([0,1,0,0,1,0])

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)



model.fit(X,y)

new_applicant = nm.array([[590,0.48,98]]) 

prediction = model.predict(new_applicant)

print(f"The prediction : {prediction[0]}")



