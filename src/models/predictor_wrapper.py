import numpy as nm
from sklearn.linear_model import LogisticRegression

class creditpredictor :
    def __init__(self,model):
        self.model = model
    
    def predict_structured(self,x_scaled):
        raw_pred = self.model.predict(x_scaled)[0]
        probabilities = self.model.predict_proba(x_scaled)[0]

        if raw_pred == 0:
            status = "Approved"
            confidence = float(probabilities[raw_pred])        
        else:
            status = "Denied"
            confidence = float(probabilities[raw_pred])

        return {
            "status" : status,
            "confidence" : confidence,
            "raw_prediction" : int(raw_pred)
        }    


if __name__ == "__main__":
    x_train = nm.array([[-1.5],[1.5],[-0.8],[0.9]])
    y_train = nm.array([0,1,0,1])

    toy_model = LogisticRegression()
    toy_model.fit(x_train,y_train)

    predictor = creditpredictor(toy_model)

    test_applicant_scaled=nm.array([[1.8]])

    result_payload = predictor.predict_structured(test_applicant_scaled)

    for key,val in result_payload.items():
        print(f"{key.capitalize()}: {val}")