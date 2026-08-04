"""
Zepto Data Engineering Pipeline - Web Scraper Module
Extracts product information from e-commerce product pages using Requests & BeautifulSoup.
Includes robust error handling, rate limiting, and mock HTML generation fallback for testing.
"""

import random
import time
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup

from data_pipeline.config import (
    DEFAULT_TARGET_ITEMS,
    MOCK_SCRAPE_URL,
    REQUEST_TIMEOUT,
    SCRAPER_HEADERS,
)
from data_pipeline.utils import ExtractionError, get_timestamp, logger


class ProductScraper:
    """Scrapes e-commerce catalog items into structured records."""

    CATEGORIES = [
        "Fruits & Vegetables",
        "Dairy & Bakery",
        "Snacks & Munchies",
        "Beverages",
        "Instant Food",
        "Personal Care",
        "Cleaning & Household",
    ]

    BRANDS = [
        "Zepto Fresh",
        "Amul",
        "Nestle",
        "Britannia",
        "Coca-Cola",
        "Tata Sampann",
        "Dabur",
        "Epigamia",
        "Lays",
    ]

    def __init__(self, target_url: str = MOCK_SCRAPE_URL):
        self.target_url = target_url
        self.session = requests.Session()
        self.session.headers.update(SCRAPER_HEADERS)

    def fetch_page(self, url: str) -> str:
        """Fetches raw HTML page content with exception handling."""
        logger.info(f"Initiating HTTP request to: {url}")
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as err:
            logger.warning(f"Live web fetch failed ({err}). Switching to dynamic HTML generator.")
            return self.generate_mock_html(num_items=DEFAULT_TARGET_ITEMS)

    def parse_html(self, html_content: str) -> List[Dict[str, Optional[any]]]:
        """Parses HTML DOM tree using BeautifulSoup to extract product details."""
        soup = BeautifulSoup(html_content, "html.parser")
        product_cards = soup.find_all("div", class_="product-card")
        
        products: List[Dict[str, Optional[any]]] = []
        for card in product_cards:
            try:
                name_elem = card.find("h3", class_="product-name")
                price_elem = card.find("span", class_="product-price")
                cat_elem = card.find("span", class_="product-category")
                brand_elem = card.find("span", class_="product-brand")
                disc_elem = card.find("span", class_="product-discount")
                rating_elem = card.find("span", class_="product-rating")
                stock_elem = card.find("span", class_="product-stock")
                url_elem = card.find("a", class_="product-link")
                img_elem = card.find("img", class_="product-img")

                product_record = {
                    "product_name": name_elem.text if name_elem else None,
                    "category": cat_elem.text if cat_elem else None,
                    "brand": brand_elem.text if brand_elem else None,
                    "price": price_elem.text if price_elem else None,
                    "discount": disc_elem.text if disc_elem else "0%",
                    "rating": rating_elem.text if rating_elem else "4.0",
                    "stock_status": stock_elem.text if stock_elem else "In Stock",
                    "product_url": url_elem["href"] if url_elem and "href" in url_elem.attrs else None,
                    "image_url": img_elem["src"] if img_elem and "src" in img_elem.attrs else None,
                    "timestamp": get_timestamp(),
                }
                products.append(product_record)
            except Exception as e:
                logger.error(f"Error parsing product card: {e}")
                continue

        logger.info(f"Successfully extracted {len(products)} raw product records.")
        return products

    def scrape(self, num_items: int = DEFAULT_TARGET_ITEMS) -> List[Dict[str, Optional[any]]]:
        """Executes full scraping flow."""
        try:
            html = self.fetch_page(self.target_url)
            products = self.parse_html(html)
            
            # Inject small artificial anomalies/missing values for transformer robustness demonstration
            if len(products) >= 5:
                products[0]["price"] = "-99.00"  # Invalid price anomaly
                products[1]["product_name"] = None  # Missing name anomaly
                products[2]["price"] = "₹ 1,450.00"  # Uncleaned currency symbol

            return products[:num_items]
        except Exception as e:
            raise ExtractionError(f"Scraping operation failed: {str(e)}") from e

    def generate_mock_html(self, num_items: int = 50) -> str:
        """Generates realistic e-commerce HTML layout for reproducible offline scraping."""
        cards = []
        sample_names = [
            ("Fresh Alphonso Mangoes", "Fruits & Vegetables"),
            ("Organic Brown Eggs 6s", "Dairy & Bakery"),
            ("A2 Cow Milk 1L", "Dairy & Bakery"),
            ("Hydrating Coconut Water 500ml", "Beverages"),
            ("Multigrain Bread 400g", "Dairy & Bakery"),
            ("Dark Chocolate Bar 100g", "Snacks & Munchies"),
            ("Premium Basmati Rice 5kg", "Instant Food"),
            ("Cold Pressed Mustard Oil 1L", "Instant Food"),
            ("Greek Yogurt Blueberry 150g", "Dairy & Bakery"),
            ("Green Tea Lemon Mint 25 bags", "Beverages"),
        ]

        for i in range(1, num_items + 1):
            base_name, base_cat = sample_names[i % len(sample_names)]
            name = f"{base_name} #{i}"
            brand = self.BRANDS[i % len(self.BRANDS)]
            price = round(random.uniform(25.0, 950.0), 2)
            discount = f"{random.choice([0, 5, 10, 15, 20, 25])}%"
            rating = round(random.uniform(3.5, 5.0), 1)
            stock = "In Stock" if i % 7 != 0 else "Out of Stock"
            prod_id = f"ZEP-{1000 + i}"

            card_html = f"""
            <div class="product-card" id="prod-{i}">
                <a class="product-link" href="https://zepto.com/p/{prod_id}">
                    <img class="product-img" src="https://images.zepto.internal/{prod_id}.jpg" alt="{name}"/>
                </a>
                <h3 class="product-name">{name}</h3>
                <span class="product-category">{base_cat}</span>
                <span class="product-brand">{brand}</span>
                <span class="product-price">₹{price:.2f}</span>
                <span class="product-discount">{discount}</span>
                <span class="product-rating">{rating}</span>
                <span class="product-stock">{stock}</span>
            </div>
            """
            cards.append(card_html)

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Zepto Catalog</title></head>
        <body>
            <div class="catalog-grid">
                {''.join(cards)}
            </div>
        </body>
        </html>
        """
        return full_html


if __name__ == "__main__":
    scraper = ProductScraper()
    items = scraper.scrape(num_items=10)
    print(f"Scraped {len(items)} items. First item preview:")
    print(items[0])
