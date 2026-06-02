#!/usr/bin/env python3
"""
Menyard Fashion Scraper - Main Orchestrator.

Pipeline:
1. Scrape all products from all collections via Shopify JSON API
2. Classify products: new / changed / unchanged (vs existing DB records)
3. Generate image + text embeddings only for new or changed products
4. Batch upsert (50/request) with retry logic
5. Mark seen products (update created_at = last_seen)
6. Delete stale products (unseen for 2+ consecutive runs)
7. Print summary
"""

import logging
import time

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

EMBEDDING_DELAY = 0.5  # seconds between HuggingFace API calls


def run_pipeline(
    skip_existing: bool = True,
    max_products: int | None = None,
) -> None:
    """Run the full smart scraping pipeline."""
    logger.info("=" * 60)
    logger.info("MENYARD SCRAPER PIPELINE - Starting")
    logger.info("=" * 60)

    db = SupabaseClient()

    # ── Step 1: Scrape ──
    logger.info("\n📦 Step 1: Scraping products from Menyard...")
    scraper = MenyardScraper()
    all_products = scraper.scrape_all()
    products_list = list(all_products.values())
    logger.info("Total unique products scraped: %d", len(products_list))

    if not products_list:
        logger.warning("No products found. Exiting.")
        return

    if max_products:
        products_list = products_list[:max_products]
        logger.info("Limited to %d products for this run.", max_products)

    # ── Step 2: Fetch existing + classify ──
    logger.info("\n🔍 Step 2: Fetching existing products & classifying...")
    existing_products = db.get_existing_products_by_source()
    seen_ids: set[str] = set()

    new_products: list[dict] = []
    changed_products: list[dict] = []
    unchanged_products: list[dict] = []

    # Track which products need image embedding regeneration
    products_needing_image_embed: list[dict] = []

    for product in products_list:
        pid = product["id"]
        seen_ids.add(pid)
        existing = existing_products.get(pid)

        if existing is None:
            # New product — needs everything
            new_products.append(product)
            products_needing_image_embed.append(product)
        elif skip_existing and not db.has_changed(product, existing):
            # Exists and unchanged — skip completely
            unchanged_products.append(product)
        else:
            # Exists but changed — needs update
            changed_products.append(product)
            if db.needs_image_embedding(product, existing):
                products_needing_image_embed.append(product)

    # ── Step 3: Generate embeddings ──
    embedder = None
    if new_products or changed_products:
        logger.info("\n🧠 Step 3: Loading SigLIP embedding model...")
        embedder = SiglipEmbedder()

    # Image embeddings (only for products that need them)
    if products_needing_image_embed and embedder:
        logger.info(
            "\n🖼️  Generating image embeddings for %d products...",
            len(products_needing_image_embed),
        )
        image_urls = [p.get("image_url") for p in products_needing_image_embed]
        valid_urls = []
        valid_indices = []

        for i, url in enumerate(image_urls):
            if url:
                valid_urls.append(url)
                valid_indices.append(i)

        if valid_urls:
            # Add staggered delay between batches
            for batch_start in range(0, len(valid_urls), 8):
                batch_end = min(batch_start + 8, len(valid_urls))
                batch_urls = valid_urls[batch_start:batch_end]
                batch_indices = valid_indices[batch_start:batch_end]

                image_embeds = embedder.embed_images(batch_urls)
                for idx, embed in zip(batch_indices, image_embeds):
                    products_needing_image_embed[idx]["image_embedding"] = embed

                success = sum(1 for e in image_embeds if e is not None)
                logger.debug(
                    "Image embedding batch %d-%d: %d/%d OK",
                    batch_start, batch_end - 1, success, len(batch_urls),
                )

                if batch_end < len(valid_urls):
                    time.sleep(EMBEDDING_DELAY)

            total_ok = sum(
                1 for p in products_needing_image_embed
                if p.get("image_embedding") is not None
            )
            logger.info(
                "Image embeddings: %d/%d successful",
                total_ok, len(valid_urls),
            )

    # Text embeddings (for ALL new + changed products)
    if (new_products or changed_products) and embedder:
        texts_for_embed = []
        text_product_refs = []  # (list_ref, index) pairs

        for p in new_products:
            texts_for_embed.append(p.get("_info_text", ""))
            text_product_refs.append((new_products, len(texts_for_embed) - 1))
        for p in changed_products:
            texts_for_embed.append(p.get("_info_text", ""))
            text_product_refs.append((changed_products, len(texts_for_embed) - 1))

        if texts_for_embed:
            logger.info(
                "\n📝 Generating text embeddings for %d products...",
                len(texts_for_embed),
            )
            # Batch in groups of 16 with delay between batches
            text_embeds = []
            for batch_start in range(0, len(texts_for_embed), 16):
                batch_end = min(batch_start + 16, len(texts_for_embed))
                batch = texts_for_embed[batch_start:batch_end]

                batch_results = embedder.embed_text(batch)
                text_embeds.extend(batch_results)

                if batch_end < len(texts_for_embed):
                    time.sleep(EMBEDDING_DELAY)

            # Assign text embeddings back to product dicts
            for i, embed in enumerate(text_embeds):
                if embed is not None:
                    ref_list, ref_idx = text_product_refs[i]
                    ref_list[ref_idx]["info_embedding"] = embed

            success_texts = sum(1 for e in text_embeds if e is not None)
            logger.info(
                "Text embeddings: %d/%d successful",
                success_texts, len(texts_for_embed),
            )

    # Clean up internal fields before DB insert
    to_upsert = new_products + changed_products
    for p in to_upsert:
        p.pop("_info_text", None)
        p.pop("_handle", None)

    # ── Step 4: Batch upsert ──
    logger.info("\n💾 Step 4: Batch upserting to Supabase...")
    upsert_ok, upsert_fail, failed_list = db.batch_upsert(to_upsert)
    logger.info(
        "Upsert: %d OK, %d failed",
        upsert_ok, upsert_fail,
    )

    # ── Step 5: Mark seen ──
    if seen_ids:
        logger.info("\n👁️  Step 5: Marking %d products as seen...", len(seen_ids))
        db.mark_seen(list(seen_ids))

    # ── Step 6: Delete stale ──
    if seen_ids:
        logger.info("\n🗑️  Step 6: Checking for stale products...")
        stale_deleted = db.delete_stale_products(seen_ids)
    else:
        stale_deleted = 0

    # ── Final count ──
    total_in_db = db.count_by_source()

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("📊 RUN SUMMARY")
    logger.info("=" * 60)
    logger.info("  🆕 New products added:       %d", len(new_products))
    logger.info("  📝 Products updated:         %d", len(changed_products))
    logger.info("  ⏭️  Products unchanged:       %d", len(unchanged_products))
    logger.info("  🗑️  Stale products deleted:   %d", stale_deleted)
    logger.info("  ❌ Upsert failures:          %d", upsert_fail)
    logger.info("  📦 Total in database:        %d", total_in_db)
    logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Menyard fashion store and import to Supabase",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all products ignoring change detection (re-upsert everything)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit products per collection (for testing)",
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
        logger.info("Scrape-only mode — extracting products to %s", args.output)
        scraper = MenyardScraper()
        per_collection = args.limit if args.limit else None
        all_products = scraper.scrape_all(max_products_per_collection=per_collection)
        products_list = list(all_products.values())

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
