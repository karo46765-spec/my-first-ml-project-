import numpy as nm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = nm.array([[2000], [3500], [5000], [7000], [9000]])
y = nm.array([[1000], [2500], [4000], [6500], [9000]])


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

print(f"calculated formula weights (Slope) : {model.coef_[0]}")
print(f"calculated baseline(Intercept) : {model.intercept_}")

new_applicant = nm.array([[6000], [4000]])

y_predict = model.predict(new_applicant)

print(
    f"\nNew Applicant Income predicted approved credit limit : ${y_predict[0],y_predict[1]}"
)
