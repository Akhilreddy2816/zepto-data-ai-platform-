"""
Zepto Data Engineering Pipeline - Database Layer
SQLAlchemy ORM module supporting SQLite and PostgreSQL databases.
Manages schema creation, product upserts, and pipeline log persistence.
"""

from datetime import datetime
from typing import List, Optional
import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from data_pipeline.config import DATABASE_URL
from data_pipeline.utils import DatabaseError, logger

Base = declarative_base()


class ProductModel(Base):
    """SQLAlchemy ORM model for zepto products."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    brand = Column(String(100), nullable=False, index=True)
    price = Column(Float, nullable=False)
    discount_percent = Column(Float, default=0.0)
    discounted_price = Column(Float, nullable=False)
    rating = Column(Float, default=4.0)
    stock_status = Column(String(50), default="In Stock")
    product_url = Column(Text, nullable=True)
    image_url = Column(Text, nullable=True)
    timestamp = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ETLLogModel(Base):
    """SQLAlchemy ORM model for pipeline execution audit logs."""

    __tablename__ = "etl_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), nullable=False)
    records_extracted = Column(Integer, default=0)
    records_loaded = Column(Integer, default=0)
    execution_time_sec = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)


class DatabaseManager:
    """Handles engine initialization, sessions, schema migration, and DB operations."""

    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        try:
            self.engine = create_engine(
                self.db_url,
                connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {}
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.init_db()
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database engine: {e}") from e

    def init_db(self) -> None:
        """Creates table schemas if they do not exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database tables initialized successfully on: {self.db_url}")
        except Exception as e:
            raise DatabaseError(f"Database schema initialization failed: {e}") from e

    def save_products_df(self, df: pd.DataFrame) -> int:
        """Bulk saves cleaned Pandas DataFrame products into the SQL database."""
        if df.empty:
            logger.warning("Attempted to save empty DataFrame to database.")
            return 0

        session = self.SessionLocal()
        try:
            records_added = 0
            for _, row in df.iterrows():
                product = ProductModel(
                    product_name=str(row["product_name"]),
                    category=str(row.get("category", "General")),
                    brand=str(row.get("brand", "Generic")),
                    price=float(row["price"]),
                    discount_percent=float(row.get("discount_percent", 0.0)),
                    discounted_price=float(row.get("discounted_price", row["price"])),
                    rating=float(row.get("rating", 4.0)),
                    stock_status=str(row.get("stock_status", "In Stock")),
                    product_url=str(row.get("product_url", "")) if pd.notna(row.get("product_url")) else None,
                    image_url=str(row.get("image_url", "")) if pd.notna(row.get("image_url")) else None,
                    timestamp=str(row.get("timestamp", datetime.utcnow().isoformat())),
                )
                session.add(product)
                records_added += 1

            session.commit()
            logger.info(f"Successfully committed {records_added} products to database.")
            return records_added
        except Exception as e:
            session.rollback()
            logger.error(f"Error persisting products to database: {e}")
            raise DatabaseError(f"Database insert transaction failed: {e}") from e
        finally:
            session.close()


    def fetch_products_dataframe(self) -> pd.DataFrame:
        """Reads products table directly into a Pandas DataFrame."""
        try:
            return pd.read_sql_table("products", con=self.engine)
        except Exception as e:
            logger.warning(f"Could not fetch products via read_sql_table ({e}). Returning empty DF.")
            return pd.DataFrame()

    def log_etl_execution(
        self,
        run_id: str,
        status: str,
        extracted: int = 0,
        loaded: int = 0,
        duration: float = 0.0,
        error_msg: Optional[str] = None
    ) -> None:
        """Persists pipeline execution metadata log entry."""
        session = self.SessionLocal()
        try:
            log_entry = ETLLogModel(
                run_id=run_id,
                status=status,
                records_extracted=extracted,
                records_loaded=loaded,
                execution_time_sec=round(duration, 3),
                error_message=error_msg
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log ETL execution state: {e}")
        finally:
            session.close()


if __name__ == "__main__":
    db = DatabaseManager()
    print("DB Manager initialized successfully.")
