import pandas as pd
import numpy as np

# Set random seed for identical dataset distribution
np.random.seed(42)
n_samples = 1000

# 1. Generate realistic underlying features
data = {
    'Client_ID': range(10001, 10001 + n_samples),
    'Age': np.random.randint(21, 70, size=n_samples),
    'Annual_Income_USD': np.random.randint(25000, 140000, size=n_samples),
    'Credit_Score': np.random.randint(500, 850, size=n_samples),
    'Debt_To_Income_Ratio': np.round(np.random.uniform(0.10, 0.65, size=n_samples), 2),
    'Credit_Utilization_Rate': np.round(np.random.uniform(0.05, 0.95, size=n_samples), 2)
}

df = pd.DataFrame(data)

# 2. Apply threshold logic (Below 650 = 0, 650 or higher = 1)
df['Target_Label'] = (df['Credit_Score'] >= 650).astype(int)

# Save dataset to CSV for machine learning training
df.to_csv('credit_training_data.csv', index=False)
print(f"Dataset saved! Total rows generated: {len(df)}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class CreditModelPipeline:
    def __init__(self, csv_path: "Users\Administrator\Desktop\ml\credit_training_data.csvr"):
        """Initialises data path, models, and scale components."""
        self.csv_path = csv_path
        self.df = None
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        
        # Placeholders for data splits
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.X_train_scaled, self.X_test_scaled = None, None

    def load_and_split_data(self, test_size: float = 0.2):
        """Loads dataset and splits features from target labels."""
        self.df = pd.read_csv(self.csv_path)
        
        # Isolate features (X) and target variable (y)
        # Drops non-predictive tracking ID if present in columns
        ignore_cols = ['Target_Label', 'Client_ID']
        X = self.df.drop(columns=[col for col in ignore_cols if col in self.df.columns])
        y = self.df['Target_Label']
        
        # Segment into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

    def preprocess_features(self):
        """Applies Z-score standardisation to features."""
        # Scales data to keep features on a uniform numeric standard
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def fit_model(self):
        """Trains the internal Logistic Regression class instance."""
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Model training completed successfully.")

    def run_evaluation(self):
        """Generates predictions and scores model performance metrics."""
        predictions = self.model.predict(self.X_test_scaled)
        
        # Calculate scores
        accuracy = accuracy_score(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        
        # Print metrics
        print("\n=== MODEL PERFORMANCE METRICS ===")
        print(f"Overall Accuracy Score: {accuracy:.4f}")
        print("\nConfusion Matrix Layout:")
        print(conf_matrix)
        print("\nDetailed Classification Report:")
        print(report)
        
    def predict_new_client(self, feature_list: list):
        """Utility method to run inference on custom raw values."""
        # Wrap into DataFrame to match original feature headers
        feature_df = pd.DataFrame([feature_list], columns=self.X_train.columns)
        scaled_features = self.scaler.transform(feature_df)
        prediction = self.model.predict(scaled_features)[0]
        probability = self.model.predict_proba(scaled_features)[0]
        
        return {
            'Class_Prediction': int(prediction),
            'Probability_Below_650': float(probability[0]),
            'Probability_Above_650': float(probability[1])
        }

# ==========================================
# EXECUTION WORKFLOW
# ==========================================
if __name__ == "__main__":
    # 1. Instantiate class with file pointer
    pipeline = CreditModelPipeline(csv_path='credit_training_data.csv')
    
    # 2. Run the internal automated training sequence
    pipeline.load_and_split_data()
    pipeline.preprocess_features()
    pipeline.fit_model()
    pipeline.run_evaluation()
    
    # 3. Predict custom applicant sample
    # Example: Age=35, Income=75000, Credit_Score=680, DTI=0.35, Util_Rate=0.22
    sample_client = [35, 75000, 680, 0.35, 0.22]
    result = pipeline.predict_new_client(sample_client)
    print(f"\nPrediction for sample input {sample_client}:")
    print(result)