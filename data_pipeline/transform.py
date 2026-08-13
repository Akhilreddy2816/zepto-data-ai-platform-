"""
Zepto Data Engineering Pipeline - Transformation & Data Cleaning Module
Cleans raw scraped e-commerce records using Pandas: deduplication, missing value imputation,
type conversion, invalid price filtering, and text normalization.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from data_pipeline.config import MAX_VALID_PRICE, MIN_VALID_PRICE
from data_pipeline.utils import TransformationError, clean_text_field, logger, parse_price


class DataTransformer:
    """Transforms raw dictionary records into cleaned, structured Pandas DataFrame."""

    def __init__(self, df_raw: Optional[pd.DataFrame] = None):
        self.df_raw = df_raw

    def raw_records_to_dataframe(self, records: List[Dict[str, any]]) -> pd.DataFrame:
        """Converts raw dictionary list into Pandas DataFrame."""
        if not records:
            logger.warning("Empty records provided to transformer. Returning empty DataFrame.")
            return pd.DataFrame()
        return pd.DataFrame(records)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes complete cleaning and transformation workflow."""
        if df.empty:
            logger.warning("DataFrame is empty. Skipping transformation.")
            return df

        logger.info(f"Starting transformation pipeline on {len(df)} initial raw rows.")

        # 1. Normalize Text Fields
        text_cols = ["product_name", "category", "brand", "stock_status"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(clean_text_field)

        # Replace 'None', 'nan', '' with NaN
        df.replace(["None", "nan", "NaN", ""], np.nan, inplace=True)

        # 2. Handle Missing Product Names
        before_drop = len(df)
        df.dropna(subset=["product_name"], inplace=True)
        dropped_names = before_drop - len(df)
        if dropped_names > 0:
            logger.info(f"Dropped {dropped_names} records missing product_name.")

        # 3. Clean and Parse Numerical Fields
        df["price"] = df["price"].apply(parse_price)
        
        # Parse Discount Percentage
        def parse_discount(disc: any) -> float:
            if pd.isna(disc) or disc is None:
                return 0.0
            cleaned = str(disc).replace("%", "").strip()
            try:
                val = float(cleaned)
                return val if 0.0 <= val <= 100.0 else 0.0
            except ValueError:
                return 0.0

        df["discount_percent"] = df["discount"].apply(parse_discount)

        # Parse Rating
        def parse_rating(rat: any) -> float:
            try:
                val = float(rat)
                return max(1.0, min(5.0, val))
            except (ValueError, TypeError):
                return 4.0

        df["rating"] = df["rating"].apply(parse_rating)

        # 4. Filter Invalid Prices
        invalid_prices = df[(df["price"].isna()) | (df["price"] < MIN_VALID_PRICE) | (df["price"] > MAX_VALID_PRICE)]
        if not invalid_prices.empty:
            logger.info(f"Removing {len(invalid_prices)} records with invalid prices.")
            df = df[~df.index.isin(invalid_prices.index)]

        # 5. Compute Engineered Feature: Final Discounted Price
        df["discounted_price"] = np.round(
            df["price"] * (1.0 - (df["discount_percent"] / 100.0)), 2
        )

        # 6. Fill Category and Brand Defaults
        for col, default_val in [("category", "General Merchandise"), ("brand", "Generic"), ("stock_status", "In Stock")]:
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)

        # 7. Deduplicate Records based on Product Name & Brand
        before_dedup = len(df)
        df.drop_duplicates(subset=["product_name", "brand"], keep="first", inplace=True)
        dedup_count = before_dedup - len(df)
        if dedup_count > 0:
            logger.info(f"Removed {dedup_count} duplicate product entries.")

        # Re-index
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Transformation complete. Cleaned dataset has {len(df)} records.")
        return df


if __name__ == "__main__":
    from data_pipeline.scraper import ProductScraper
    scraper = ProductScraper()
    raw_data = scraper.scrape(10)
    transformer = DataTransformer()
    df_raw = transformer.raw_records_to_dataframe(raw_data)
    df_cleaned = transformer.clean(df_raw)
    print(df_cleaned.head())
