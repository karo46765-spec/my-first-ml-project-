import numpy as nm
import yaml
from sklearn.linear_model import LogisticRegression

from src.preprocessing.data_preprocessor import creditprocessor
from src.models.predictor_wrapper import creditpredictor

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

nm.random.seed(42)
n_samples = 1000000
credit_scores = nm.random.randint(170, 1000, size=(n_samples, 1))
dti_ratios = nm.random.beta(a=2.5, b=3, size=(n_samples, 1)) * 1.2
# Merge features horizontally into a single array
x_train = nm.hstack((credit_scores, dti_ratios))

# We calculate a mathematical risk score: higher risk if Credit Score is low AND DTI is high
risk_score = ((850 - x_train[:, 0]) / 550 * 0.6) + (x_train[:, 1] * 0.4)


# Pass the risk through a sigmoid function to convert it into a probability (0 to 1)
probabilities = 1 / (1 + nm.exp(-10 * (risk_score - 0.5)))

# Generate binary outcomes (0 = Low Risk/Approve, 1 = High Risk/Default) using the probabilities
y_train = nm.random.binomial(1, probabilities)

preprocessor = creditprocessor()
preprocessor.fit_pipeline(x_train)
x_train_scaled = preprocessor.transform_data(x_train)

toy_model = LogisticRegression(random_state=config["model"]["random_state"])
toy_model.fit(x_train_scaled, y_train)

predictor = creditpredictor(toy_model)

raw_applicant = nm.array([[450, 0.75]])

preprocessor.fit_pipeline(raw_applicant)
scaledapplicant = preprocessor.transform_data(raw_applicant)

final_responce = predictor.predict_structured(scaledapplicant)


def verify(self):
    self.assertIsInstance(final_responce, dict)


print(f"{final_responce['prediction_status']}")
print(f"{final_responce['confidence_probabilities']}")
print(f"{final_responce}")
print(raw_applicant)
print(x_train)
