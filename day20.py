import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. Define raw data
x_raw = np.array([[750,0.1],[720,0.2],[680,0.3],[790,0.15],[710,0.25],[690,0.2],[690,0.2],[740,0.12],[770,0.18],[520,0.65],[490,0.7]])
y_raw = np.array([0,0,0,0,0,0,0,0,0,1,1])

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.18, random_state=42,stratify=y_raw)

print("---Original Class Distribution---")
print(f"Safe Profiles (Class 0): {np.sum(y_train == 0)}")
print(f"Fraud Profiles (Class 1): {np.sum(y_train == 1)}") # Fixed label text

# 3. Apply SMOTE
sme = SMOTE(random_state=42, k_neighbors=1) # Note: k_neighbors adjusted for small data size
x_reworked, y_reworked = sme.fit_resample(X_train, y_train)

print("\n---Balanced Class Distribution---")
print(f"Safe Profiles (Class 0): {np.sum(y_reworked == 0)}")
print(f"Fraud Profiles (Class 1): {np.sum(y_reworked == 1)}")

# Fixed: .Shape changed to .shape
print(f"\nTotal Combined Training Matrix Shape: {x_reworked.shape}")




