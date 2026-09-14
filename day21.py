import numpy as nm
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

raw_data = {
    "Credit_Score": [750, 510, 680, 790, 710, 490, 740, 770],
    "Employement_type": [
        "Salaried",
        "Unemployed",
        "Salaried",
        "Salareid",
        "Self-Employed",
        "Unemployed",
        "Salareid",
        "Self-Employed",
    ],
}

df = pd.DataFrame(raw_data)
y = nm.array([0, 1, 0, 0, 0, 1, 0, 0])

df_encoded = pd.get_dummies(df, columns=["Employement_type"]).astype(int)

sme = SMOTE(
    random_state=42, k_neighbors=1
)  # Note: k_neighbors adjusted for small data size
x_reworked, y_reworked = sme.fit_resample(df_encoded, y)

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=5,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
)

pipeline_model = model.fit(x_reworked, y_reworked)

print(
    f"Original Row Count : {len(df)}->Balanced Row count after SMOTE:{len(x_reworked)}"
)
print(
    f"Class 1 (Risk) instances before: {nm.sum(y == 1)} -> After SMOTE : {nm.sum(y_reworked == 1)}"
)
print(f"XGBOOST Pipeline Training Complete : {pipeline_model}")
