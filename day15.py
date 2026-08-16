
import numpy as nm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



X = nm.array([[750,0.2],[500,0.6],[620,0.4],[710,0.1],[580,0.5]])

y = nm.array([0,1,0,0,1])

# Reserve 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5,random_state=42)

model.fit(X_train,y_train)

new_applicant=nm.array([[890,0.1],[790,0.8],[811,0.2],[200,1]])

prediction =  model.predict(new_applicant)

print(f"The prediction {prediction[0]}\n {prediction[1]}\n {prediction[2]}\n {prediction[3]}")



