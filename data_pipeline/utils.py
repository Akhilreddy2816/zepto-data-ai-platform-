"""
Zepto Data Engineering Pipeline - Utilities & Logging
"""

import logging
import re
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "data_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger with formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


class DataPipelineError(Exception):
    """Base exception class for data engineering pipeline errors."""
    pass


class ExtractionError(DataPipelineError):
    """Raised when web scraping or data extraction fails."""
    pass


class TransformationError(DataPipelineError):
    """Raised when data transformation/cleaning validation fails."""
    pass


class DatabaseError(DataPipelineError):
    """Raised when database operations fail."""
    pass


def clean_text_field(text: Optional[str]) -> str:
    """Normalizes string inputs by stripping whitespace and removing illegal characters."""
    if text is None or not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_price(price_str: any) -> Optional[float]:
    """Extracts numeric price float from currency formatted strings."""
    if price_str is None:
        return None
    if isinstance(price_str, (int, float)):
        return float(price_str)
    
    cleaned = re.sub(r"[^\d.-]", "", str(price_str))
    try:
        return float(cleaned) if cleaned and cleaned != "-" else None
    except ValueError:
        return None


def get_timestamp() -> str:
    """Returns current ISO formatted timestamp."""
    return datetime.utcnow().isoformat()
