"""
Zepto Analytics & Machine Learning - Inference Engine
Loads best model artifact (model.pkl) and performs real-time delivery delay predictions.
"""

from pathlib import Path
from typing import Dict, List, Union
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


class DeliveryPredictor:
    """Predicts quick-commerce delivery delay risk using trained model artifact."""

    def __init__(self, model_path: str = str(MODEL_PATH)):
        self.model_path = Path(model_path)
        self.artifact = None
        self.load_model()

    def load_model(self) -> None:
        """Loads model pickle artifact containing scaler, encoders, and feature names."""
        if not self.model_path.exists():
            from analytics.train import ModelTrainer
            trainer = ModelTrainer()
            _, _, self.artifact = trainer.train_and_evaluate_all()
        else:
            self.artifact = joblib.load(self.model_path)

        self.model = self.artifact["model"]
        self.scaler = self.artifact["scaler"]
        self.label_encoders = self.artifact["label_encoders"]
        self.feature_names = self.artifact["feature_names"]

    def predict_sample(self, sample_data: Dict[str, Union[int, float, str]]) -> Dict[str, any]:
        """Predicts delay risk for a single order payload."""
        df_input = pd.DataFrame([sample_data])

        # Apply Categorical Label Encoders
        for col, le in self.label_encoders.items():
            if col in df_input.columns:
                val = str(df_input[col].iloc[0])
                if val in le.classes_:
                    df_input[col] = le.transform([val])[0]
                else:
                    df_input[col] = 0

        # Ensure correct column ordering
        df_input = df_input[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(df_input)

        # Make Prediction
        pred_class = int(self.model.predict(X_scaled)[0])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
            delay_prob = float(probs[1])
        else:
            delay_prob = 1.0 if pred_class == 1 else 0.0

        risk_level = "High" if delay_prob >= 0.6 else ("Medium" if delay_prob >= 0.35 else "Low")
        status_label = "Delayed (>15 mins)" if pred_class == 1 else "On-Time (<=15 mins)"

        return {
            "prediction_class": pred_class,
            "status_label": status_label,
            "delay_probability": round(delay_prob, 4),
            "delay_probability_percent": f"{round(delay_prob * 100, 1)}%",
            "risk_level": risk_level,
            "model_used": self.artifact["model_name"],
        }


if __name__ == "__main__":
    predictor = DeliveryPredictor()
    test_payload = {
        "customer_tenure_months": 12,
        "order_distance_km": 6.5,
        "item_count": 8,
        "order_value_inr": 1250.0,
        "traffic_density": "High",
        "weather_condition": "Rainy",
        "driver_experience_years": 2,
        "delivery_time_mins": 22.0,
    }
    result = predictor.predict_sample(test_payload)
    print("Inference Result:", result)
