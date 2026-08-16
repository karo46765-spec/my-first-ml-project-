from sklearn import model_selection
import numpy as nm 
import joblib
from sklearn.linear_model import LogisticRegression

x = nm.array([[700],[550],[600],[650]])
y = nm.array([0,1,1,0])

model = LogisticRegression()
model.fit(x,y)

model_filename = "credit_model.joblib"


# 1. Save your trained model to a file
joblib.dump(model, "credit_model.joblib")

# 2. Load the saved model back later
loaded_model = joblib.load("credit_model.joblib")

new_applicant = nm.array([[720]])
prediction = loaded_model.predict(new_applicant)

print(f"{prediction[0]}")

