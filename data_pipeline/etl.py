"""
Zepto Data Engineering Pipeline - ETL Orchestrator
Executes automated pipeline: Extract -> Clean -> Transform -> Load to SQL -> Export CSV.
"""

import time
import uuid
from typing import Dict, Tuple
import pandas as pd

from data_pipeline.config import CLEANED_DATA_PATH, RAW_DATA_PATH
from data_pipeline.database import DatabaseManager
from data_pipeline.scraper import ProductScraper
from data_pipeline.transform import DataTransformer
from data_pipeline.utils import DataPipelineError, logger


class ETLPipeline:
    """Orchestrates end-to-end Data Engineering ETL workflow."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.scraper = ProductScraper()
        self.transformer = DataTransformer()

    def run(self, num_items: int = 50) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Executes full ETL flow and returns (cleaned_df, execution_summary)."""
        run_id = f"ETL-{uuid.uuid4().hex[:8].upper()}"
        start_time = time.time()
        logger.info(f"=== Starting Zepto ETL Pipeline Run: {run_id} ===")

        try:
            # 1. EXTRACT
            raw_records = self.scraper.scrape(num_items=num_items)
            extracted_count = len(raw_records)
            
            # Export Raw CSV
            df_raw = self.transformer.raw_records_to_dataframe(raw_records)
            df_raw.to_csv(RAW_DATA_PATH, index=False)
            logger.info(f"Raw scraped data saved to: {RAW_DATA_PATH}")

            # 2. TRANSFORM & CLEAN
            df_cleaned = self.transformer.clean(df_raw)

            # Export Cleaned CSV
            df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
            logger.info(f"Cleaned product data exported to CSV: {CLEANED_DATA_PATH}")

            # 3. LOAD TO DATABASE
            loaded_count = self.db.save_products_df(df_cleaned)

            duration = time.time() - start_time
            summary = {
                "run_id": run_id,
                "status": "SUCCESS",
                "extracted_records": extracted_count,
                "cleaned_records": len(df_cleaned),
                "loaded_records": loaded_count,
                "duration_seconds": round(duration, 3),
                "raw_csv": str(RAW_DATA_PATH),
                "cleaned_csv": str(CLEANED_DATA_PATH),
            }

            # 4. LOG AUDIT
            self.db.log_etl_execution(
                run_id=run_id,
                status="SUCCESS",
                extracted=extracted_count,
                loaded=loaded_count,
                duration=duration
            )

            logger.info(f"=== ETL Pipeline Run {run_id} Completed Successfully in {duration:.2f}s ===")
            return df_cleaned, summary

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"=== ETL Pipeline Run {run_id} Failed: {error_msg} ===")
            
            self.db.log_etl_execution(
                run_id=run_id,
                status="FAILED",
                duration=duration,
                error_msg=error_msg
            )
            raise DataPipelineError(f"Pipeline execution failed: {error_msg}") from e


def run_etl_pipeline(num_items: int = 50) -> Dict[str, any]:
    """Convenience functional interface for ETL pipeline execution."""
    pipeline = ETLPipeline()
    _, summary = pipeline.run(num_items=num_items)
    return summary


if __name__ == "__main__":
    result = run_etl_pipeline(50)
    print("ETL Summary:", result)
