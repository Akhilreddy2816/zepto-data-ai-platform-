"""
Zepto Analytics & Machine Learning - Data Visualization Module
Generates comprehensive EDA charts: Histogram, Scatter Plot, Box Plot, Heatmap,
Pair Plot, Count Plot, Pie Chart, Bar Chart, Line Chart.
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set Seaborn Zepto Theme
sns.set_theme(style="darkgrid", palette="deep")
ZEPTO_COLORS = ["#7000B8", "#FF3269", "#00C853", "#FF9100", "#29B6F6"]


class AnalyticsVisualizer:
    """Generates and exports data science visualization figures."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    def plot_histogram(self, df: pd.DataFrame, col: str = "delivery_time_mins") -> plt.Figure:
        """1. Histogram of numerical distributions."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col], kde=True, color="#7000B8", ax=ax, bins=25)
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col.replace('_', ' ').title())
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"histogram_{col}.png")
        return fig

    def plot_scatter(
        self, df: pd.DataFrame, x_col: str = "order_distance_km", y_col: str = "delivery_time_mins"
    ) -> plt.Figure:
        """2. Scatter plot demonstrating correlation between 2 metrics."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue="is_delayed" if "is_delayed" in df.columns else None,
            palette=["#7000B8", "#FF3269"], alpha=0.7, ax=ax
        )
        ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"scatter_{x_col}_{y_col}.png")
        return fig

    def plot_boxplot(self, df: pd.DataFrame, col: str = "delivery_time_mins", by: str = "traffic_density") -> plt.Figure:
        """3. Box plot to detect outliers across categorical groups."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=by, y=col, palette="Purples", ax=ax)
        ax.set_title(f"{col.replace('_', ' ').title()} across {by.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"boxplot_{col}_{by}.png")
        return fig

    def plot_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """4. Heatmap showing numerical feature correlation matrix."""
        fig, ax = plt.subplots(figsize=(9, 7))
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", ax=ax, cbar=True)
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "heatmap_correlation.png")
        return fig

    def plot_pairplot(self, df: pd.DataFrame) -> sns.PairGrid:
        """5. Pair plot analyzing feature interactions."""
        cols = ["order_distance_km", "item_count", "order_value_inr", "delivery_time_mins", "is_delayed"]
        subset = df[[c for c in cols if c in df.columns]]
        grid = sns.pairplot(subset, hue="is_delayed" if "is_delayed" in subset.columns else None, palette=["#7000B8", "#FF3269"])
        grid.fig.suptitle("Pair Plot Analysis", y=1.02, fontsize=14, fontweight="bold")
        grid.savefig(PLOTS_DIR / "pairplot_matrix.png")
        return grid

    def plot_countplot(self, df: pd.DataFrame, col: str = "traffic_density") -> plt.Figure:
        """6. Count plot for categorical value counts."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=df, x=col, palette="rocket", ax=ax)
        ax.set_title(f"Order Volume by {col.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"countplot_{col}.png")
        return fig

    def plot_piechart(self, df: pd.DataFrame, col: str = "weather_condition") -> plt.Figure:
        """7. Pie chart showing category proportions."""
        fig, ax = plt.subplots(figsize=(6, 6))
        counts = df[col].value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=ZEPTO_COLORS[:len(counts)], startangle=140)
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"pie_{col}.png")
        return fig

    def plot_barchart(self, df: pd.DataFrame, x_col: str = "traffic_density", y_col: str = "delivery_time_mins") -> plt.Figure:
        """8. Bar chart comparing mean values."""
        fig, ax = plt.subplots(figsize=(8, 5))
        df_grouped = df.groupby(x_col)[y_col].mean().reset_index()
        sns.barplot(data=df_grouped, x=x_col, y=y_col, palette="Purples_r", ax=ax)
        ax.set_title(f"Mean {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"bar_{x_col}_{y_col}.png")
        return fig

    def plot_linechart(self, df: pd.DataFrame) -> plt.Figure:
        """9. Line chart visualizing metric trends across tenure or sample sequence."""
        fig, ax = plt.subplots(figsize=(9, 5))
        df_sorted = df.sort_values("customer_tenure_months")
        grouped = df_sorted.groupby("customer_tenure_months")["order_value_inr"].mean().reset_index()
        sns.lineplot(data=grouped, x="customer_tenure_months", y="order_value_inr", marker="o", color="#FF3269", linewidth=2.5, ax=ax)
        ax.set_title("Customer Tenure vs Average Order Value (INR)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Customer Tenure (Months)")
        ax.set_ylabel("Avg Order Value (INR)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "line_tenure_order_value.png")
        return fig

    def generate_all_plots(self, df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """Generates and saves all 9 visual EDA charts."""
        plots = {
            "histogram": self.plot_histogram(df),
            "scatter": self.plot_scatter(df),
            "boxplot": self.plot_boxplot(df),
            "heatmap": self.plot_heatmap(df),
            "countplot": self.plot_countplot(df),
            "piechart": self.plot_piechart(df),
            "barchart": self.plot_barchart(df),
            "linechart": self.plot_linechart(df),
        }
        self.plot_pairplot(df)
        print(f"Generated and saved all 9 EDA plots in: {PLOTS_DIR}")
        return plots


if __name__ == "__main__":
    from analytics.preprocessing import DataPreprocessor
    proc = DataPreprocessor()
    df = proc.clean_and_impute(proc.load_data())
    viz = AnalyticsVisualizer()
    viz.generate_all_plots(df)
