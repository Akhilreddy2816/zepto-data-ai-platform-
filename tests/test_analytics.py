"""
pytest Unit Tests for Module 2: Data Analytics & Machine Learning
"""

import numpy as np
import pandas as pd
import pytest
from analytics.preprocessing import DataPreprocessor
from analytics.train import ModelTrainer
from analytics.predict import DeliveryPredictor


def test_synthetic_data_generation():
    processor = DataPreprocessor()
    df = processor.generate_synthetic_dataset(num_samples=100)
    assert len(df) == 100
    assert "is_delayed" in df.columns
    assert "delivery_time_mins" in df.columns


def test_preprocessing_pipeline():
    processor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feats = processor.prepare_train_test_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feats) > 0


def test_model_training_and_inference():
    trainer = ModelTrainer()
    leaderboard, best_name, artifact = trainer.train_and_evaluate_all()
    assert len(leaderboard) >= 4
    assert best_name in leaderboard

    predictor = DeliveryPredictor()
    sample = {
        "customer_tenure_months": 12,
        "order_distance_km": 5.0,
        "item_count": 4,
        "order_value_inr": 450.0,
        "traffic_density": "Low",
        "weather_condition": "Clear",
        "driver_experience_years": 5,
        "delivery_time_mins": 14.0,
    }
    result = predictor.predict_sample(sample)
    assert "prediction_class" in result
    assert "delay_probability" in result
    assert result["prediction_class"] in [0, 1]
