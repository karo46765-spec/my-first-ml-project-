import unittest
import unittest
import numpy as nm
from sklearn.linear_model import LogisticRegression
from day27 import creditpredictor

class testcreditpredictor(unittest.TestCase):
    def setUp(self):
        x_toy = nm.array([[-2.0],[-1.0],[1.0],[2.0]])
        y_toy = nm.array([0,0,1,1])

        self.toy_model = LogisticRegression()
        self.toy_model.fit(x_toy,y_toy)

        self.predictor = creditpredictor(self.toy_model)

    def test_output(self):
        mock_scaled = nm.array([[-5.0]])
        payload = self.predictor.predict_structured(mock_scaled)

        self.assertIsInstance(payload,dict)

        self.assertIn("prediction_status",payload)
        self.assertEqual(payload["prediction_status"],"Approved")

        self.assertIn("confidence_probabilities",payload)
        self.assertIsInstance(payload["confidence_probabilities"], dict)
        approved_score = payload["confidence_probabilities"]["Approved"]
        self.assertIsInstance(approved_score, float)       
        self.assertGreater(approved_score,0.5)

if __name__ == "__main__":
    unittest.main()     

