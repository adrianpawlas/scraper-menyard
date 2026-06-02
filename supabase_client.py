"""Supabase client for product database operations.

Features:
- Batch upserts (50 products per request)
- Change detection via content_hash (stored in `other` field as JSON)
- Stale product cleanup (delete after 2+ consecutive missed runs)
- Retry with exponential backoff for failed batches
- Failure logging to failed_products.log
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, SOURCE

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_RETRIES = 3
STALE_DAYS = 7  # ~2 runs (Tue/Fri = 3-4 days apart)

FAILURE_LOG = "failed_products.log"


def compute_content_hash(product_data: dict) -> str:
    """Compute a hash of the product content fields for change detection."""
    content = {
        "title": product_data.get("title"),
        "price": product_data.get("price"),
        "sale": product_data.get("sale"),
        "description": product_data.get("description"),
        "image_url": product_data.get("image_url"),
        "additional_images": product_data.get("additional_images"),
        "category": product_data.get("category"),
        "size": product_data.get("size"),
        "tags": product_data.get("tags"),
    }
    raw = json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.md5(raw).hexdigest()


def build_record(product_data: dict) -> dict:
    """Convert product dict into a DB record suitable for upsert."""
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

    record = {}
    for our_key, db_col in field_map.items():
        value = product_data.get(our_key)
        if value is not None:
            record[db_col] = value

    # Always set created_at to now (acts as last_seen timestamp)
    record["created_at"] = datetime.now(timezone.utc).isoformat()

    # Store content_hash in the `other` field as JSON
    content_hash = compute_content_hash(product_data)
    other_data = {"content_hash": content_hash}
    record["other"] = json.dumps(other_data, ensure_ascii=False)

    # Convert numpy arrays → lists for pgvector
    for emb_field in ("image_embedding", "info_embedding"):
        val = record.get(emb_field)
        if isinstance(val, np.ndarray):
            record[emb_field] = val.tolist()

    return record


def log_failed_products(products: list[dict], error: str) -> None:
    """Append failed product info to the failure log file."""
    try:
        with open(FAILURE_LOG, "a") as f:
            timestamp = datetime.now(timezone.utc).isoformat()
            f.write(f"\n--- {timestamp} | Error: {error} ---\n")
            for p in products:
                f.write(f"  {p.get('id', '?')} | {p.get('title', '?')} | {p.get('product_url', '?')}\n")
    except Exception as e:
        logger.warning("Failed to write failure log: %s", e)


class SupabaseClient:
    """Client for interacting with the Supabase products table."""

    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ── Fetch existing products ──────────────────────────────────────

    def get_existing_products_by_source(self) -> dict[str, dict]:
        """Fetch all products for this source with fields needed for change detection.

        Returns dict mapping product id → { title, price, sale, image_url, ... }
        """
        try:
            response = (
                self.client.table(SUPABASE_TABLE)
                .select("id, title, price, sale, description, image_url, "
                        "additional_images, category, size, tags, other, "
                        "image_embedding")
                .eq("source", SOURCE)
                .execute()
            )
            products: dict[str, dict] = {}
            for row in response.data:
                pid = row.pop("id")
                products[pid] = row
            logger.info("Fetched %d existing products from DB", len(products))
            return products
        except Exception as e:
            logger.warning("Failed to fetch existing products: %s", e)
            return {}

    # ── Change detection ─────────────────────────────────────────────

    def has_changed(self, scraped: dict, existing: dict) -> bool:
        """Compare scraped product against existing DB record.

        Returns True if any content field has changed.
        """
        scraped_hash = compute_content_hash(scraped)

        # Parse stored hash from `other` field
        other_raw = existing.get("other")
        stored_hash = None
        if other_raw:
            try:
                other_data = json.loads(other_raw) if isinstance(other_raw, str) else {}
                stored_hash = other_data.get("content_hash")
            except (json.JSONDecodeError, TypeError):
                pass

        if stored_hash is None:
            # No stored hash — treat as changed
            return True

        return scraped_hash != stored_hash

    # ── Batch upsert with retry ──────────────────────────────────────

    def batch_upsert(self, products: list[dict]) -> tuple[int, int, list[dict]]:
        """Upsert products in batches of BATCH_SIZE.

        Returns (success_count, fail_count, failed_products_list).
        Retries failed batches up to MAX_RETRIES times.
        """
        if not products:
            return 0, 0, []

        success = 0
        fail = 0
        failed_products: list[dict] = []

        for batch_start in range(0, len(products), BATCH_SIZE):
            batch = products[batch_start:batch_start + BATCH_SIZE]
            # Build records once per batch (not per retry attempt)
            records = [build_record(p) for p in batch]

            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    self.client.table(SUPABASE_TABLE).upsert(
                        records,
                        on_conflict="source, product_url",
                    ).execute()
                    success += len(batch)
                    logger.debug(
                        "Batch upserted %d products (batch %d)",
                        len(batch), batch_start // BATCH_SIZE,
                    )
                    break  # success
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Batch upsert attempt %d/%d failed for %d products: %s",
                        attempt + 1, MAX_RETRIES, len(batch), e,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        fail += len(batch)
                        failed_products.extend(batch)
                        logger.error(
                            "Batch upsert failed after %d attempts for %d products",
                            MAX_RETRIES, len(batch),
                        )
                        log_failed_products(batch, str(last_error))

        return success, fail, failed_products

    # ── Update created_at (last_seen) for seen products ─────────────

    def mark_seen(self, product_ids: list[str]) -> int:
        """Update created_at for all seen products so they're not marked stale.

        Returns count of updated records.
        """
        if not product_ids:
            return 0

        updated = 0
        # Process in batches to avoid overly large queries
        for i in range(0, len(product_ids), BATCH_SIZE):
            batch_ids = product_ids[i:i + BATCH_SIZE]
            now = datetime.now(timezone.utc).isoformat()
            try:
                self.client.table(SUPABASE_TABLE)\
                    .update({"created_at": now})\
                    .eq("source", SOURCE)\
                    .in_("id", batch_ids)\
                    .execute()
                updated += len(batch_ids)
            except Exception as e:
                logger.warning("Failed to mark_seen batch: %s", e)

        logger.info("Marked %d products as seen", updated)
        return updated

    # ── Delete stale products ────────────────────────────────────────

    def delete_stale_products(self, seen_ids: set[str]) -> int:
        """Delete products from this source that weren't seen in this run
        AND haven't been seen for STALE_DAYS (i.e., missed 2+ consecutive runs).

        Returns count of deleted products.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)).isoformat()

        try:
            # Find products not in the seen set
            # Supabase REST doesn't support NOT IN with a large list directly,
            # so we use: source=SOURCE AND created_at < cutoff AND id NOT IN (...)
            # But for efficiency, we first fetch all IDs for this source,
            # then compute the stale set in Python, then delete.
            response = (
                self.client.table(SUPABASE_TABLE)
                .select("id, created_at")
                .eq("source", SOURCE)
                .lt("created_at", cutoff)
                .execute()
            )

            stale_ids = []
            for row in response.data:
                if row["id"] not in seen_ids:
                    stale_ids.append(row["id"])

            if not stale_ids:
                logger.info("No stale products to delete")
                return 0

            # Delete in batches
            deleted = 0
            for i in range(0, len(stale_ids), BATCH_SIZE):
                batch = stale_ids[i:i + BATCH_SIZE]
                try:
                    self.client.table(SUPABASE_TABLE)\
                        .delete()\
                        .eq("source", SOURCE)\
                        .in_("id", batch)\
                        .execute()
                    deleted += len(batch)
                except Exception as e:
                    logger.warning("Failed to delete stale batch: %s", e)

            logger.info("Deleted %d stale products (not seen for %d+ days)",
                        deleted, STALE_DAYS)
            return deleted
        except Exception as e:
            logger.warning("Failed to query stale products: %s", e)
            return 0

    # ── Convenience: check if product needs image embedding ──────────

    def needs_image_embedding(self, product: dict, existing: dict | None) -> bool:
        """Check if a product needs a new image embedding.

        Returns True if:
        - Product is new (not in DB)
        - Product's image_url changed
        - Product has no existing image_embedding
        """
        if existing is None:
            return True

        # Check if image_url changed
        if product.get("image_url") != existing.get("image_url"):
            return True

        # Check if existing has no embedding
        if not existing.get("image_embedding"):
            return True

        return False

    # ── Count products by source ─────────────────────────────────────

    def count_by_source(self) -> int:
        """Count total products for this source in the database."""
        try:
            response = (
                self.client.table(SUPABASE_TABLE)
                .select("id", count="exact")
                .eq("source", SOURCE)
                .execute()
            )
            return response.count or 0
        except Exception:
            return 0
