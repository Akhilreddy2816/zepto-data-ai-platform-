"""
Zepto Data Engineering Pipeline - Configuration Module
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Scraping Configuration
MOCK_SCRAPE_URL = "https://zepto-mock-store.internal/products"
DEFAULT_TARGET_ITEMS = 50
SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 10  # seconds

# Database Configuration
DEFAULT_DB_PATH = DATA_DIR / "zepto_products.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Output Paths
RAW_DATA_PATH = DATA_DIR / "raw_products.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_products.csv"

# Product Validation Constraints
MIN_VALID_PRICE = 1.0
MAX_VALID_PRICE = 5000.0
VALID_CATEGORIES = [
    "Fruits & Vegetables",
    "Dairy & Bakery",
    "Snacks & Munchies",
    "Beverages",
    "Instant Food",
    "Personal Care",
    "Cleaning & Household",
]
