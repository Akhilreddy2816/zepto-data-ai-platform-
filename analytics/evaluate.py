"""
Zepto Analytics & Machine Learning - Evaluation & Feature Importance Engine
Computes ROC curves, confusion matrices, feature importance graphs, and metrics breakdowns.
"""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)

from analytics.preprocessing import DataPreprocessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelEvaluator:
    """Evaluates trained models and exports evaluation diagnostic plots."""

    def __init__(self, model_path: str = str(MODEL_PATH)):
        self.model_path = Path(model_path)
        self.artifact = None
        if self.model_path.exists():
            self.artifact = joblib.load(self.model_path)

    def evaluate_model(self) -> Dict[str, any]:
        """Runs full evaluation on the stored best model artifact."""
        if not self.artifact:
            from analytics.train import ModelTrainer
            trainer = ModelTrainer()
            _, _, self.artifact = trainer.train_and_evaluate_all()

        model = self.artifact["model"]
        feature_names = self.artifact["feature_names"]

        preprocessor = DataPreprocessor()
        _, X_test, _, y_test, _ = preprocessor.prepare_train_test_data()

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        # Plot Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax_cm,
                    xticklabels=["On Time", "Delayed"], yticklabels=["On Time", "Delayed"])
        ax_cm.set_title(f"Confusion Matrix ({self.artifact['model_name']})", fontsize=14, fontweight="bold")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        fig_cm.tight_layout()
        fig_cm.savefig(PLOTS_DIR / "confusion_matrix.png")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
        ax_roc.plot(fpr, tpr, color="#7000B8", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"Receiver Operating Characteristic - {self.artifact['model_name']}", fontsize=13, fontweight="bold")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        fig_roc.savefig(PLOTS_DIR / "roc_curve.png")

        # Plot Feature Importance
        feature_importance_df = self.plot_feature_importance(model, feature_names)

        return {
            "model_name": self.artifact["model_name"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm.tolist(),
            "feature_importance": feature_importance_df.to_dict(orient="records"),
            "fig_cm": fig_cm,
            "fig_roc": fig_roc,
        }

    def plot_feature_importance(self, model: any, feature_names: list) -> float:
        """Plots and exports feature importance graph for tree-based models or coefficients."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)

        importances_df = (
            importances_df := np.array(importances)
        )
        
        df_imp = pd.DataFrame({
            "Feature": [f.replace("_", " ").title() for f in feature_names],
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df_imp, x="Importance", y="Feature", palette="Purples_r", ax=ax_imp)
        ax_imp.set_title("Feature Importance Graph", fontsize=14, fontweight="bold")
        fig_imp.tight_layout()
        fig_imp.savefig(PLOTS_DIR / "feature_importance.png")
        plt.close(fig_imp)

        return df_imp


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model()
    print("Evaluation Results:", {k: v for k, v in results.items() if not k.startswith("fig_")})
