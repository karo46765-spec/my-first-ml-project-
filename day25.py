
import joblib

from pandas.core.common import random_state

import numpy as nm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV , cross_val_score
import joblib

x = nm.array([[750,0.1],[720,0.2],[680,0.3],[790,0.15],[710,0.25],[690,0.2],[690,0.2],[740,0.12],[770,0.18],[520,0.65],[490,0.7]])
y = nm.array([0,0,0,0,0,0,0,0,0,1,1])

base_model = XGBClassifier(random_state=42)

param_grid={
    'n_estimators' : [5,15],
    'learning_rate' : [0.1,0.3]
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=2,                      # 5-fold cross-validation
    scoring='accuracy',         # Evaluation metric
    n_jobs=-1,                 # Use all available CPU cores
    verbose=1                  # Print progress logs
)


best_model = grid_search.fit(x,y)

best_model_filename="bestmodel.joblib"

joblib.dump(best_model,"bestmodel.joblib")

loded_model = joblib.load("bestmodel.joblib")

scores = cross_val_score(base_model, x, y, cv=2, scoring='accuracy')

print("Best Hyperparameters:", loded_model.best_params_)
print("Best Cross-Validation Score:", loded_model.best_score_)
print(loded_model.best_estimator_)

print(f"{scores.mean() * 100}")

print(f"{loded_model}")