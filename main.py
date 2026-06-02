#!/usr/bin/env python3
"""
Menyard Fashion Scraper - Main Orchestrator.

Pipeline:
1. Scrape all products from all collections via Shopify JSON API
2. Download product images and generate image embeddings (SigLIP 768-dim)
3. Generate text embeddings from product info (SigLIP 768-dim)
4. Upsert everything into Supabase
"""

import logging
import sys
import time
from datetime import datetime, timezone

from tqdm import tqdm

from config import SOURCE
from scraper import MenyardScraper
from embeddings import SiglipEmbedder
from supabase_client import SupabaseClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run_pipeline(
    skip_existing: bool = True,
    max_products: int | None = None,
) -> None:
    """Run the full scraping pipeline."""
    logger.info("=" * 60)
    logger.info("MENYARD SCRAPER PIPELINE - Starting")
    logger.info("=" * 60)

    # ── Step 1: Scrape products ──
    logger.info("\n📦 Step 1: Scraping products from Menyard...")
    scraper = MenyardScraper()
    all_products = scraper.scrape_all()

    products_list = list(all_products.values())
    logger.info("Total unique products scraped: %d", len(products_list))

    if not products_list:
        logger.warning("No products found. Exiting.")
        return

    # Optionally limit for testing
    if max_products:
        products_list = products_list[:max_products]
        logger.info("Limited to %d products for this run.", max_products)

    # ── Step 1b: Check existing products in DB ──
    if skip_existing:
        logger.info("\n🔍 Checking existing products in Supabase...")
        db = SupabaseClient()
        existing_ids = db.get_existing_ids()
        products_to_process = [
            p for p in products_list if p["id"] not in existing_ids
        ]
        skipped = len(products_list) - len(products_to_process)
        logger.info(
            "Products: %d total, %d existing (skipped), %d to process",
            len(products_list), skipped, len(products_to_process),
        )
        products_list = products_to_process
        if not products_list:
            logger.info("All products already in database. Done!")
            return

    # ── Step 2: Generate embeddings ──
    logger.info("\n🧠 Step 2: Loading SigLIP embedding model...")
    embedder = SiglipEmbedder()

    # Image embeddings
    logger.info("\n🖼️  Generating image embeddings...")
    image_urls = [p.get("image_url") for p in products_list]
    valid_urls = []
    valid_indices = []

    for i, url in enumerate(image_urls):
        if url:
            valid_urls.append(url)
            valid_indices.append(i)

    if valid_urls:
        image_embeds = embedder.embed_images(valid_urls)
        for idx, embed in zip(valid_indices, image_embeds):
            products_list[idx]["image_embedding"] = embed
        success_images = sum(1 for e in image_embeds if e is not None)
        logger.info(
            "Image embeddings: %d/%d successful",
            success_images, len(valid_urls),
        )

    # Text embeddings
    logger.info("\n📝 Generating text embeddings...")
    info_texts = [p.get("_info_text", "") for p in products_list]
    text_embeds = embedder.embed_text(info_texts)
    for i, embed in enumerate(text_embeds):
        products_list[i]["info_embedding"] = embed
    success_texts = sum(1 for e in text_embeds if e is not None)
    logger.info(
        "Text embeddings: %d/%d successful",
        success_texts, len(info_texts),
    )

    # Clean up internal fields before DB insert
    for p in products_list:
        p.pop("_info_text", None)
        p.pop("_handle", None)

    # ── Step 3: Upsert to Supabase ──
    logger.info("\n💾 Step 3: Upserting to Supabase...")
    db = SupabaseClient()
    success_count = 0
    fail_count = 0

    for product in tqdm(products_list, desc="Upserting"):
        if db.upsert_product(product):
            success_count += 1
        else:
            fail_count += 1

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total products scraped:    %d", len(products_list))
    logger.info("Successfully upserted:     %d", success_count)
    logger.info("Failed to upsert:          %d", fail_count)
    logger.info("Image embeddings:          %d/%d", success_images if valid_urls else 0, len(valid_urls) if valid_urls else 0)
    logger.info("Text embeddings:           %d/%d", success_texts, len(info_texts))
    logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Menyard fashion store and import to Supabase"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all products, including existing ones (re-upsert)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of products to process (for testing)",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape products, save to JSON (skip embeddings & DB)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="products_menyard.json",
        help="Output JSON file path (for --scrape-only)",
    )
    args = parser.parse_args()

    if args.scrape_only:
        logger.info("Scrape-only mode - extracting products to %s", args.output)
        scraper = MenyardScraper()
        # Use limit as per-collection limit for faster testing
        per_collection = args.limit if args.limit else None
        all_products = scraper.scrape_all(max_products_per_collection=per_collection)
        products_list = list(all_products.values())

        # Clean internal fields
        for p in products_list:
            p.pop("_info_text", None)
            p.pop("_handle", None)

        import json

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(products_list, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d products to %s", len(products_list), args.output)
        return

    run_pipeline(
        skip_existing=not args.all,
        max_products=args.limit,
    )


if __name__ == "__main__":
    main()
