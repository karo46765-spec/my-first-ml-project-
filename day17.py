from sklearn.ensemble import RandomForestClassifier


import numpy as nm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

data = {
    "Credit_score": [750, 500, 620, 710, 580, 690],
    "D_to_I ratio": [0.2, 0.6, 0.4, 0.1, 0.5, 0.3],
    "Age": [45, 23, 34, 52, 29, 41],
}

x = pd.DataFrame(data)
y = nm.array([0, 1, 0, 0, 1, 0])


forest_model = RandomForestClassifier(
    n_estimators=50,  # Number of decision trees in the forest
    max_depth=None,  # Max depth of each tree (None means grow until pure)
    min_samples_split=2,  # Min samples required to split an internal node
    random_state=42,  # Seed for consistent bootstrapping and feature selection
    n_jobs=-1,
)


forest_model.fit(x, y)

importance = forest_model.feature_importances_

importances = pd.Series(importance, index=x.columns)

importance_df = pd.DataFrame({"Feature": x.columns, "Importance_Score": importance})

Sorted_importance = importance_df.sort_values(by="Importance_Score", ascending=False)

print("----Feature Importance Ranking-----")
print(Sorted_importance)
