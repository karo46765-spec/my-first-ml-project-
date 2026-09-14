from sklearn.model_selection import permutation_test_score
import numpy as nm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

x = nm.array(
    [
        [750, 0.1],
        [720, 0.2],
        [680, 0.3],
        [790, 0.15],
        [710, 0.25],
        [690, 0.2],
        [690, 0.2],
        [740, 0.12],
        [770, 0.18],
        [520, 0.65],
        [490, 0.7],
    ]
)
y = nm.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

model = RandomForestClassifier(n_estimators=10, random_state=42)

scores = cross_val_score(model, x, y, cv=3, scoring="accuracy")

print(f"Scores for each independent fold : {scores}")
print(f"Mean Reliability Accuracy :{scores.mean() * 100:.2f}")
