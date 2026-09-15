from locust import FastHttpUser, task, between


class CreditScoringUser(FastHttpUser):
    wait_time = between(0.2, 0.8)

    @task(4)
    def test_predict_endpoint(self):
        """Simulate high-volume credit risk prediction"""
        payload = {"credit_score": 720.0, "dti_ratio": 0.25}
        self.client.post("/predict", json=payload)

    @task(1)
    def test_health_endpoint(self):
        """Simulate periodic infrastructure health"""
        self.client.get("/health")

    @task(1)
    def test_readiness_check(self):
        """simulate periodic readiness probe"""
        self.client.get("/ready")
