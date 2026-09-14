import numpy as nm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

x = nm.array([[750, 0.2], [500, 0.6], [620, 0.4], [710, 0.1], [580, 0.5], [690, 0.3]])
y = nm.array([0, 1, 0, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.18, random_state=42
)


base_model = RandomForestClassifier(random_state=42)


print("automated parameter grid search>>>")

param_grid = {
    "n_estimators": [50, 100, 200, 10],
    "max_depth": [10, 20, 3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=2,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
test_accuracy = best_rf_model.score(X_test, y_test)
print("Test Set Accuracy:", test_accuracy)
