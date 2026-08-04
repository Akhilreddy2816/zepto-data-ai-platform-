# Module 1: Data Engineering Pipeline

Automated E-Commerce Data Engineering Pipeline for Zepto product catalog data collection, cleaning, normalization, SQL storage, and CSV export.

## Architecture

```
Raw Web Source / Mock HTML Engine
            │
            ▼ (Requests & BeautifulSoup)
       scraper.py
            │
            ▼ (Raw Records)
      transform.py (Pandas text normalization, deduplication & price validation)
            │
            ▼ (Cleaned DataFrame)
      database.py (SQLAlchemy ORM -> SQLite / PostgreSQL)
            │
            ▼
   cleaned_products.csv & Database Tables
```

## Module Structure

- `scraper.py`: ProductScraper class extracting e-commerce HTML DOM elements with HTTP headers, rate limiting, and mock HTML generator fallback.
- `transform.py`: DataTransformer class enforcing text normalization, missing name filtering, numerical parsing, discount calculations, and deduplication.
- `database.py`: DatabaseManager handling SQLAlchemy ORM models (`products` and `etl_logs`), transaction management, and connection pooling.
- `etl.py`: ETLPipeline orchestrator connecting Extract, Transform, Load, logging, and CSV export.
- `config.py`: Centralized configuration settings and file paths.
- `utils.py`: Structured logging, custom exception classes, string parsing utilities.

## How to Run Standalone

```bash
python -m data_pipeline.etl
```
