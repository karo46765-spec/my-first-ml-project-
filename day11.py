from day10 import new_applicant
import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.array([[540], [590], [630], [710], [780]])

Y_train = np.array([1, 1, 1, 0, 0])

Classifier = LogisticRegression()

Classifier.fit(X_train, Y_train)

new_applicant = np.array([[789], [567], [900], [999]])

prediction = Classifier.predict(new_applicant)

probabilities = Classifier.predict_proba(new_applicant)

print(
    f"The prediction of creditscorwe and there percentage :{prediction},{probabilities}"
)
