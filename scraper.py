"""Shopify scraper for Menyard product data.

Uses the Shopify JSON API to:
1. Fetch all product handles from each collection (paginated)
2. Fetch full product details for each product
3. Extract and structure all required fields
"""

from __future__ import annotations

import json
import logging
import re as _re
import time
from typing import Any

import requests

from config import (
    COLLECTIONS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    SHOPIFY_PAGE_LIMIT,
    SHOP_URL,
    BASE_CURRENCY,
    SOURCE,
    BRAND,
    GENDER,
    SECOND_HAND,
)
from image_utils import filter_product_images

logger = logging.getLogger(__name__)

# Approximate EUR→USD rate for multi-currency price display
EUR_USD_RATE = 1.08


def _format_price_str(price_eur: float | None) -> str | None:
    """Format price with EUR primary + USD secondary.

    Shopify API returns EUR. We add USD at approximate rate.
    Format: "20.90EUR,22.57USD"
    """
    if price_eur is None:
        return None
    eur_str = f"{price_eur:.2f}{BASE_CURRENCY}"
    usd_str = f"{price_eur * EUR_USD_RATE:.2f}USD"
    return f"{eur_str},{usd_str}"


class MenyardScraper:
    """Scrapes product data from Menyard's Shopify store."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })

    def _request_with_retry(self, url: str) -> dict | None:
        """Make a GET request with retry logic.

        Respects Retry-After header for 429 rate limits.
        """
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                resp = e.response
                if resp is not None and resp.status_code == 404:
                    logger.warning("404 on %s", url)
                    return None

                if resp is not None and resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = int(retry_after)
                    else:
                        wait = 10
                    logger.warning(
                        "Rate limited (429) on %s. Waiting %ds (attempt %d/%d).",
                        url, wait, attempt + 1, MAX_RETRIES,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                        continue
                    return None

                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt + 1, MAX_RETRIES, url, e,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt + 1, MAX_RETRIES, url, e,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def get_collection_products(self, handle: str) -> list[dict]:
        """Get all products from a collection via paginated products.json endpoint.

        Returns list of partial product dicts (from collection listing).
        """
        products = []
        page = 1

        while True:
            url = (
                f"{SHOP_URL}/collections/{handle}/products.json"
                f"?page={page}&limit={SHOPIFY_PAGE_LIMIT}"
            )
            logger.info("Fetching collection %s page %d", handle, page)

            data = self._request_with_retry(url)
            if data is None:
                break

            batch = data.get("products", [])
            if not batch:
                break

            products.extend(batch)
            logger.info(
                "Got %d products from %s page %d (total: %d)",
                len(batch), handle, page, len(products),
            )

            if len(batch) < SHOPIFY_PAGE_LIMIT:
                break

            page += 1
            time.sleep(0.3)

        return products

    def get_product_detail(self, handle: str) -> dict | None:
        """Fetch full product details from /products/{handle}.json."""
        url = f"{SHOP_URL}/products/{handle}.json"
        data = self._request_with_retry(url)
        if data is None:
            return None
        return data.get("product")

    def extract_product_data(
        self, product: dict, collection_categories: list[str],
    ) -> dict:
        """Convert raw Shopify product JSON into our normalized format."""
        handle = product["handle"]
        product_id = f"menyard_{product['id']}"

        # --- Title ---
        title = product.get("title", "").strip()

        # --- Description (strip HTML) ---
        description = product.get("body_html") or ""
        if description:
            description = _re.sub(r"<[^>]+>", " ", description)
            description = _re.sub(r"\s+", " ", description).strip()

        # --- Tags & Category ---
        tags = product.get("tags", "")
        if isinstance(tags, str):
            tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
        else:
            tag_list = [t.strip().lower() for t in tags if t.strip()]

        category_parts = list(collection_categories) if collection_categories else []

        category_tag_map = {
            "t-shirt": "T-Shirts", "tshirt": "T-Shirts",
            "sweater": "Sweaters", "hoodie": "Hoodies",
            "jacket": "Jackets", "jeans": "Jeans",
            "pants": "Pants", "shorts": "Shorts",
            "shirt": "Shirts", "polo": "Polos",
            "beanie": "Beanies", "cap": "Caps", "hat": "Hats",
            "bag": "Bags", "tote": "Totes",
            "socks": "Socks", "boxer": "Boxers",
            "belt": "Belts", "glasses": "Glasses",
            "accessories": "Accessories", "accessoire": "Accessories",
        }
        for tag in tag_list:
            for key, cat in category_tag_map.items():
                if key in tag and cat not in category_parts:
                    category_parts.append(cat)

        category = ", ".join(category_parts) if category_parts else None

        # --- Prices (multi-currency: EUR + USD) ---
        variants = product.get("variants", [])
        if not variants:
            price_str = None
            sale_str = None
        else:
            prices_eur = []
            compare_prices = []
            for v in variants:
                p = v.get("price")
                if p:
                    try:
                        prices_eur.append(float(p))
                    except (ValueError, TypeError):
                        pass
                cp = v.get("compare_at_price")
                if cp:
                    try:
                        compare_prices.append(float(cp))
                    except (ValueError, TypeError):
                        pass

            if prices_eur:
                min_price = min(prices_eur)
                if compare_prices:
                    min_compare = min(compare_prices)
                    sale_str = _format_price_str(min_price)
                    price_str = _format_price_str(min_compare)
                else:
                    price_str = _format_price_str(min_price)
                    sale_str = None
            else:
                price_str = None
                sale_str = None

        # --- Product URL ---
        product_url = f"{SHOP_URL}/products/{handle}"

        # --- Sizes ---
        sizes = None
        for opt in product.get("options", []):
            if opt.get("name", "").lower() == "size":
                values = opt.get("values", [])
                if values:
                    sizes = ", ".join(values)
                break

        # --- Images (filtered) ---
        images = product.get("images", [])
        main_image, additional_images = filter_product_images(
            images, product_title=title,
        )
        additional_images_str = (
            " , ".join(additional_images) if additional_images else None
        )

        # --- Metadata (all info in one JSON field) ---
        metadata = {
            "shopify_id": product["id"],
            "handle": handle,
            "title": title,
            "description": description,
            "category": category,
            "price": price_str,
            "sale": sale_str,
            "sizes": sizes,
            "vendor": product.get("vendor"),
            "product_type": product.get("product_type", ""),
            "tags": tags,
            "collections": collection_categories,
            "gender": GENDER,
            "brand": BRAND,
        }
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        # --- Text for info_embedding ---
        info_text_parts = [
            f"Title: {title}",
            f"Brand: {BRAND}",
            f"Gender: {GENDER}",
            f"Category: {category}" if category else "",
            f"Price: {price_str}" if price_str else "",
            f"Sale: {sale_str}" if sale_str else "",
            f"Sizes: {sizes}" if sizes else "",
            f"Description: {description}" if description else "",
            f"Tags: {tags}" if tags else "",
        ]
        info_text = ". ".join(p for p in info_text_parts if p)

        return {
            "id": product_id,
            "source": SOURCE,
            "product_url": product_url,
            "affiliate_url": None,
            "image_url": main_image,
            "brand": BRAND,
            "title": title,
            "description": description,
            "category": category,
            "gender": GENDER,
            "created_at": None,
            "metadata": metadata_json,
            "size": sizes,
            "second_hand": SECOND_HAND,
            "image_embedding": None,
            "country": None,
            "compressed_image_url": None,
            "tags": tag_list,
            "search_vector": None,
            "title_tsv": None,
            "brand_tsv": None,
            "description_tsv": None,
            "other": None,
            "price": price_str,
            "sale": sale_str,
            "additional_images": additional_images_str,
            "info_embedding": None,
            "_info_text": info_text,
            "_handle": handle,
        }

    def scrape_all(
        self,
        max_products_per_collection: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Scrape all products from all collections.

        Args:
            max_products_per_collection: Limit products per collection (for testing).

        Returns:
            Dict mapping product_id → product_data (deduplicated across collections).
        """
        all_products: dict[str, dict[str, Any]] = {}
        product_categories: dict[str, set] = {}

        for collection in COLLECTIONS:
            handle = collection["handle"]
            base_categories = collection.get("categories")

            logger.info("=" * 60)
            logger.info(
                "Scraping collection: %s (%s)", collection["label"], handle,
            )
            logger.info("=" * 60)

            partial_products = self.get_collection_products(handle)
            if max_products_per_collection:
                partial_products = partial_products[:max_products_per_collection]

            logger.info(
                "Found %d products in %s", len(partial_products), handle,
            )

            for partial in partial_products:
                pid = f"menyard_{partial['id']}"
                prod_handle = partial["handle"]

                detail = self.get_product_detail(prod_handle)
                if detail is None:
                    logger.warning("Failed to fetch detail for %s", prod_handle)
                    continue

                if base_categories:
                    if pid not in product_categories:
                        product_categories[pid] = set()
                    product_categories[pid].update(base_categories)

                if pid not in all_products:
                    categories_for_product = list(product_categories.get(pid, []))
                    processed = self.extract_product_data(
                        detail, categories_for_product,
                    )
                    all_products[pid] = processed

                time.sleep(0.1)

        # Final category update: merge collection membership with tag-derived categories
        for pid, product_data in all_products.items():
            if pid in product_categories:
                collection_cats = set(product_categories[pid])
                # Preserve any tag-derived categories already set by extract_product_data
                existing = product_data.get("category", "")
                if existing:
                    for ec in existing.split(", "):
                        ec = ec.strip()
                        if ec:
                            collection_cats.add(ec)
                product_data["category"] = ", ".join(sorted(collection_cats))

        return all_products
