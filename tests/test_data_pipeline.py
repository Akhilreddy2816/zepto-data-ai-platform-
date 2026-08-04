"""
pytest Unit Tests for Module 1: Data Engineering Pipeline
"""

import pandas as pd
import pytest
from data_pipeline.database import DatabaseManager
from data_pipeline.scraper import ProductScraper
from data_pipeline.transform import DataTransformer
from data_pipeline.utils import parse_price, clean_text_field


def test_parse_price():
    assert parse_price("₹ 450.00") == 450.00
    assert parse_price(12.5) == 12.5
    assert parse_price(None) is None
    assert parse_price("invalid") is None


def test_clean_text_field():
    assert clean_text_field("  Zepto   Fresh  ") == "Zepto Fresh"
    assert clean_text_field(None) == ""


def test_scraper_mock_html():
    scraper = ProductScraper()
    items = scraper.scrape(num_items=10)
    assert len(items) > 0
    assert "product_name" in items[0]
    assert "price" in items[0]


def test_transformer_clean():
    transformer = DataTransformer()
    raw_sample = [
        {"product_name": "  Amul Butter  ", "price": "₹ 250.00", "discount": "10%", "rating": "4.5"},
        {"product_name": None, "price": "100"},  # Should be dropped
        {"product_name": "Bad Price", "price": "-50.0"},  # Should be dropped
    ]
    df_raw = pd.DataFrame(raw_sample)
    df_clean = transformer.clean(df_raw)
    assert len(df_clean) == 1
    assert df_clean.iloc[0]["product_name"] == "Amul Butter"
    assert df_clean.iloc[0]["price"] == 250.0
    assert df_clean.iloc[0]["discounted_price"] == 225.0


def test_database_manager(tmp_path):
    db_file = tmp_path / "test_zepto.db"
    db = DatabaseManager(db_url=f"sqlite:///{db_file}")
    
    test_df = pd.DataFrame([{
        "product_name": "Test Milk",
        "category": "Dairy",
        "brand": "Amul",
        "price": 50.0,
        "discount_percent": 0.0,
        "discounted_price": 50.0,
        "rating": 4.5,
        "stock_status": "In Stock",
        "product_url": "http://test.com",
        "image_url": "http://test.com/img.jpg",
        "timestamp": "2026-07-25T00:00:00"
    }])
    
    inserted = db.save_products_df(test_df)
    assert inserted == 1
    
    fetched = db.fetch_products_dataframe()
    assert len(fetched) == 1
    assert fetched.iloc[0]["product_name"] == "Test Milk"
