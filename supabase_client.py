"""Supabase client for inserting/upserting products into the database."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Client for interacting with the Supabase products table."""

    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def upsert_product(self, product_data: dict[str, Any]) -> bool:
        """Insert or update a product record.

        Uses the unique constraint (source, product_url) for upsert.

        Returns True if successful, False otherwise.
        """
        # Build the record for insertion
        record = {}

        # Map our fields to database columns
        field_map = {
            "id": "id",
            "source": "source",
            "product_url": "product_url",
            "affiliate_url": "affiliate_url",
            "image_url": "image_url",
            "brand": "brand",
            "title": "title",
            "description": "description",
            "category": "category",
            "gender": "gender",
            "created_at": "created_at",
            "metadata": "metadata",
            "size": "size",
            "second_hand": "second_hand",
            "image_embedding": "image_embedding",
            "country": "country",
            "compressed_image_url": "compressed_image_url",
            "tags": "tags",
            "search_vector": "search_vector",
            "title_tsv": "title_tsv",
            "brand_tsv": "brand_tsv",
            "description_tsv": "description_tsv",
            "other": "other",
            "price": "price",
            "sale": "sale",
            "additional_images": "additional_images",
            "info_embedding": "info_embedding",
        }

        for our_key, db_col in field_map.items():
            value = product_data.get(our_key)
            if value is not None:
                record[db_col] = value

        # Set created_at to current time
        record["created_at"] = datetime.now(timezone.utc).isoformat()

        # Handle numpy arrays → list of floats for pgvector
        if "image_embedding" in record and isinstance(record["image_embedding"], np.ndarray):
            record["image_embedding"] = record["image_embedding"].tolist()

        if "info_embedding" in record and isinstance(record["info_embedding"], np.ndarray):
            record["info_embedding"] = record["info_embedding"].tolist()

        try:
            # Use upsert with the unique constraint on (source, product_url)
            response = (
                self.client.table(SUPABASE_TABLE)
                .upsert(
                    record,
                    on_conflict="source, product_url",
                )
                .execute()
            )
            logger.debug("Upserted product: %s", product_data.get("title", "unknown"))
            return True
        except Exception as e:
            logger.error(
                "Failed to upsert product %s (%s): %s",
                product_data.get("title", "unknown"),
                product_data.get("id", "unknown"),
                e,
            )
            return False

    def upsert_products_batch(self, products: list[dict[str, Any]]) -> tuple[int, int]:
        """Upsert multiple products.

        Returns (success_count, fail_count).
        """
        success = 0
        fail = 0
        for product in products:
            if self.upsert_product(product):
                success += 1
            else:
                fail += 1
        return success, fail

    def get_existing_ids(self) -> set[str]:
        """Get all existing product IDs in the database to skip re-processing."""
        try:
            response = (
                self.client.table(SUPABASE_TABLE)
                .select("id, image_embedding")
                .eq("source", "scraper-menyard")
                .execute()
            )
            existing = set()
            for row in response.data:
                existing.add(row["id"])
            logger.info("Found %d existing products in database", len(existing))
            return existing
        except Exception as e:
            logger.warning("Failed to fetch existing products: %s", e)
            return set()
