"""
Zepto Analytics & Machine Learning - Preprocessing Engine
Handles data loading, synthetic delivery/customer dataset generation, missing value imputation,
outlier detection & capping, categorical encoding, and feature scaling.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "zepto_delivery_analytics.csv"


class DataPreprocessor:
    """Preprocesses raw analytics data into model-ready features."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else DATA_PATH
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def generate_synthetic_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generates a realistic Zepto Quick-Commerce Order & Delivery Performance Dataset."""
        np.random.seed(42)

        customer_tenure_months = np.random.randint(1, 36, size=num_samples)
        order_distance_km = np.round(np.random.uniform(0.5, 8.5, size=num_samples), 2)
        item_count = np.random.randint(1, 15, size=num_samples)
        order_value_inr = np.round(item_count * np.random.uniform(80, 250, size=num_samples) + np.random.uniform(10, 50, size=num_samples), 2)
        traffic_density = np.random.choice(["Low", "Medium", "High"], size=num_samples, p=[0.3, 0.5, 0.2])
        weather_condition = np.random.choice(["Clear", "Rainy", "Foggy"], size=num_samples, p=[0.7, 0.2, 0.1])
        driver_experience_years = np.random.randint(1, 10, size=num_samples)

        # Base delivery time calculation with noise
        base_time = (order_distance_km * 2.5) + (item_count * 0.8) + np.random.normal(5, 2, size=num_samples)
        base_time += np.where(traffic_density == "High", 7.0, np.where(traffic_density == "Medium", 3.0, 0.0))
        base_time += np.where(weather_condition == "Rainy", 6.0, np.where(weather_condition == "Foggy", 4.0, 0.0))
        delivery_time_mins = np.round(np.clip(base_time, 8.0, 45.0), 1)

        # Target: Delayed Delivery (> 15 minutes benchmark for quick-commerce)
        is_delayed = (delivery_time_mins > 16.5).astype(int)

        df = pd.DataFrame({
            "customer_tenure_months": customer_tenure_months,
            "order_distance_km": order_distance_km,
            "item_count": item_count,
            "order_value_inr": order_value_inr,
            "traffic_density": traffic_density,
            "weather_condition": weather_condition,
            "driver_experience_years": driver_experience_years,
            "delivery_time_mins": delivery_time_mins,
            "is_delayed": is_delayed,
        })

        # Inject minor null values to test imputation logic
        null_idx = np.random.choice(df.index, size=20, replace=False)
        df.loc[null_idx, "driver_experience_years"] = np.nan

        # Save to disk
        df.to_csv(self.data_path, index=False)
        print(f"Generated synthetic Zepto dataset with {num_samples} samples -> {self.data_path}")
        return df

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from disk or generates it if absent."""
        if not self.data_path.exists():
            return self.generate_synthetic_dataset()
        return pd.read_csv(self.data_path)

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans dataset and imputes missing values."""
        df_clean = df.copy()
        
        # Deduplicate
        df_clean.drop_duplicates(inplace=True)

        # Numerical Imputation (Median)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val if pd.notna(median_val) else 0.0)

        # Categorical Imputation (Mode)
        cat_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                modes = df_clean[col].mode()
                mode_val = modes[0] if not modes.empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(mode_val)

        return df_clean

    def handle_outliers(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Caps numerical outliers using IQR range clipping."""
        df_out = df.copy()
        if columns is None:
            columns = ["order_value_inr", "delivery_time_mins"]

        for col in columns:
            if col in df_out.columns and np.issubdtype(df_out[col].dtype, np.number):
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

        return df_out

    def encode_and_scale(
        self, df: pd.DataFrame, target_col: str = "is_delayed"
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Encodes categorical variables, scales numerical features, returns X, y."""
        df_proc = df.copy()

        # Separate target
        if target_col not in df_proc.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")

        y = df_proc[target_col].values
        df_features = df_proc.drop(columns=[target_col])

        # Encode Categorical Features
        cat_cols = df_features.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
            self.label_encoders[col] = le

        # Fill any remaining NaNs in features
        df_features.fillna(0, inplace=True)

        self.feature_names = list(df_features.columns)

        # Scale Features
        X_scaled = self.scaler.fit_transform(df_features)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_scaled, y, self.feature_names

    def prepare_train_test_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Executes full preprocessing pipeline and splits data into Train/Test sets."""
        raw_df = self.load_data()
        cleaned_df = self.clean_and_impute(raw_df)
        capped_df = self.handle_outliers(cleaned_df)
        X, y, feature_names = self.encode_and_scale(capped_df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    processor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feats = processor.prepare_train_test_data()
    print(f"Dataset successfully prepared. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features ({len(feats)}): {feats}")
