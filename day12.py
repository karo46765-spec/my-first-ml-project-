import numpy as nm
from sklearn.metrics import confusion_matrix , classification_report

y_true = nm.array([0,0,0,1,1,1])
y_pred = nm.array([0,1,0,1,0,1])

cm = confusion_matrix(y_true, y_pred)
print(cm)

report = classification_report(y_true,y_pred)
print(report)

print("--- Confusion Matrix Array ---")
print(cm)

print("\n ---Detailed Report---")
print(report)

tn,fp,fn,tp = cm.ravel()

print("---Breakdown Anaylisis---")
print(f"Safe profiles correctly caught (True Negatives) : {tn}")
print(f"Risk profiles correctly caught (True positives) : {tp}")

print(f"Dangerous misses (false negatives - risk labeled as safe) : {fn}")

