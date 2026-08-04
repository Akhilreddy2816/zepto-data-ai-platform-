# Module 2: Data Analytics & Machine Learning

Data Science and Predictive Machine Learning Pipeline for Zepto delivery performance, order delay risk, and customer analytics.

## Workflow

```
Load Dataset
    │
    ▼
Preprocessing & Imputation (preprocessing.py)
    │
    ▼
Exploratory Analysis & Plots (visualize.py - 9 chart types)
    │
    ▼
Model Training & 5-Fold Cross Validation (train.py)
    ├─ Logistic Regression
    ├─ Decision Tree
    ├─ Random Forest
    ├─ Gradient Boosting
    └─ XGBoost
    │
    ▼
Evaluation & Feature Importance (evaluate.py)
    │
    ▼
Best Model Pickling (model.pkl) & Inference (predict.py)
```

## Structure

- `preprocessing.py`: Synthetic dataset generator, median/mode imputation, IQR outlier capping, label encoding, standard scaling.
- `visualize.py`: Generates 9 visualizations (Histogram, Scatter, Box plot, Heatmap, Pair plot, Count plot, Pie chart, Bar chart, Line chart).
- `train.py`: Model benchmarks across 5 classifiers with 5-fold Stratified Cross Validation and metric comparison leaderboard.
- `evaluate.py`: Generates Confusion Matrix heatmaps, ROC curves, and Feature Importance bar charts.
- `predict.py`: Loads `model.pkl` and provides batch/single prediction APIs.
- `eda.ipynb`: Interactive Jupyter Notebook executing step-by-step EDA and ML benchmark workflow.

## How to Run Standalone

```bash
# Train models and export model.pkl
python -m analytics.train

# Evaluate model metrics and plots
python -m analytics.evaluate

# Run sample inference
python -m analytics.predict
```
