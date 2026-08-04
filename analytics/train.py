"""
Zepto Analytics & Machine Learning - Model Training Engine
Trains, cross-validates, compares ML classifiers: Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting, and XGBoost. Exports the top-performing model artifact.
"""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from analytics.preprocessing import DataPreprocessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Attempt XGBoost import with fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class ModelTrainer:
    """Trains and compares classification algorithms on Zepto delivery dataset."""

    def __init__(self):
        self.preprocessor = DataPreprocessor()

    def get_candidate_models(self) -> Dict[str, any]:
        """Instantiates classifier algorithms."""
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        }

        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric="logloss", random_state=42
            )
        else:
            models["Extra Trees (XGB Fallback)"] = ExtraTreesClassifier(n_estimators=100, random_state=42)

        return models

    def train_and_evaluate_all(self) -> Tuple[Dict[str, Dict[str, float]], str, any]:
        """Trains models, performs cross-validation, evaluates metrics, selects and exports best model."""
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.prepare_train_test_data()

        models = self.get_candidate_models()
        leaderboard: Dict[str, Dict[str, float]] = {}
        trained_instances = {}

        best_score = -1.0
        best_model_name = ""
        best_model_obj = None

        print("=== Training & Evaluating ML Models ===")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            # Cross Validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
            mean_cv_f1 = float(np.mean(cv_scores))

            # Fit model on training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, y_proba))
            else:
                auc = 0.0

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))

            leaderboard[name] = {
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "ROC AUC": round(auc, 4),
                "CV F1 Mean": round(mean_cv_f1, 4),
            }

            trained_instances[name] = model

            print(f"[{name}] -> Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f} | CV F1: {mean_cv_f1:.4f}")

            # Select Best Model based on F1 Score / ROC AUC
            composite_score = f1 + (0.5 * auc)
            if composite_score > best_score:
                best_score = composite_score
                best_model_name = name
                best_model_obj = model

        # Package best model with metadata & preprocessor details
        artifact = {
            "model_name": best_model_name,
            "model": best_model_obj,
            "scaler": self.preprocessor.scaler,
            "label_encoders": self.preprocessor.label_encoders,
            "feature_names": feature_names,
            "metrics": leaderboard[best_model_name],
        }

        # Export Model Pickles
        joblib.dump(artifact, MODEL_PATH)
        joblib.dump(artifact, MODELS_DIR / "best_model.joblib")
        print(f"\n[BEST MODEL] Best Model Selected: {best_model_name} (Composite Score: {best_score:.4f})")
        print(f"Saved trained artifact to: {MODEL_PATH}")

        return leaderboard, best_model_name, artifact


if __name__ == "__main__":
    trainer = ModelTrainer()
    results, best_name, _ = trainer.train_and_evaluate_all()
    print("\nModel Leaderboard:")
    for k, v in results.items():
        print(f"{k}: {v}")
