from sklearn.linear_model import LogisticRegression
import numpy as nm
import pandas as pd

from src.preprocessing.data_preprocessor import creditprocessor
from src.models.predictor_wrapper import creditpredictor
from src.exception import DataValidationError

x_train = nm.array([[700, 0.2], [550, 0.6], [600, 0.4]])
y_train = nm.array([0, 1, 0])

preproccessor = creditprocessor()
preproccessor.fit_pipeline(x_train)

x_scaled = preproccessor.transform_data(x_train)

toy_model = LogisticRegression(random_state=43)
toy_model.fit(x_scaled, y_train)

predictor = creditpredictor(toy_model)

print("Sample One")
try:
    raw_applicant = nm.array([[450, 0.7]])
    scaled_applicant = preproccessor.transform_data(raw_applicant)
    final_responce = predictor.predict_structured(scaled_applicant)

except DataValidationError as e:
    print(f"'{e}'")

print("Sample Two")
try:
    invalid_applicant = nm.array([450, 0.75])
    desc_invalid = preproccessor.transform_data(invalid_applicant)
except DataValidationError as e:
    print(f"'{e}'")


print(f"{final_responce['prediction_status']}")
print(f"{final_responce}")
