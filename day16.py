from day11 import prediction
from day10 import new_applicant
import numpy as nm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = nm.array(
    [
        [750, 0.2, 45],
        [500, 0.6, 23],
        [620, 0.4, 34],
        [710, 0.1, 52],
        [580, 0.5, 29],
        [690, 0.3, 41],
    ]
)

y = nm.array([0, 1, 0, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


forest_model = RandomForestClassifier(
    n_estimators=50,  # Number of decision trees in the forest
    max_depth=None,  # Max depth of each tree (None means grow until pure)
    min_samples_split=2,  # Min samples required to split an internal node
    random_state=42,  # Seed for consistent bootstrapping and feature selection
    n_jobs=-1,  # Use all available CPU cores for faster training
)

forest_model.fit(X_train, y_train)

new_applicant = nm.array(
    [
        [610, 0.45, 31],
        [590, 0.34, 23],
        [689, 0.80, 34],
        [790, 0.1, 52],
        [510, 0.5, 50],
        [690, 0.9, 80],
    ]
)

prediction = forest_model.predict(new_applicant)

print(
    f"\nNew applicant prediction :{prediction[0]}{prediction[1]},{prediction[2]},{prediction[3]},{prediction[4]},{prediction[5]}"
)
