# Menyard Fashion Scraper

Scrapes all products from [Menyard Homme](https://www.menyardhomme.com) — a Shopify-based fashion store — and imports them into Supabase with SigLIP image and text embeddings.

## Features

- **Shopify API scraper** – Uses `/collections/{handle}/products.json` (paginated) and `/products/{handle}.json`
- **Smart image filtering** – Distinguishes clean product-on-background shots from lifestyle/cropped/model shots using filename heuristics
- **SigLIP embeddings** – 768-dim image and text embeddings via `google/siglip-base-patch16-384`
- **Supabase upsert** – Uses the `(source, product_url)` unique constraint for idempotent inserts

## Requirements

- Python 3.10+
- PyTorch (see [pytorch.org](https://pytorch.org) for install instructions)
- Supabase project with a `products` table (schema provided)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Full pipeline (scrape → embed → import)

```bash
python main.py
```

### Scrape only (save to JSON, no embeddings or DB)

```bash
python main.py --scrape-only --output products.json
```

### Limit products for testing

```bash
python main.py --limit 10
```

### Re-process all products (including existing)

```bash
python main.py --all
```

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Constants and configuration |
| `scraper.py` | Shopify API scraper with pagination |
| `image_utils.py` | Image URL filtering heuristics |
| `embeddings.py` | SigLIP model wrapper |
| `supabase_client.py` | Supabase upsert operations |
| `main.py` | Pipeline orchestrator + CLI |

## How It Works

1. **Scrape** – For each collection, paginates through `/collections/{handle}/products.json?page=N&limit=250`.
2. **Collect** – Fetches full product detail from `/products/{handle}.json`.
3. **Filter** – Rejects "bad" images (cropped, e-commerce batch exports, numbered prefixes, date patterns, overly descriptive names).
4. **Embed** – Downloads product images and runs them through SigLIP for 768-dim image embeddings. Also generates text embeddings from product metadata.
5. **Import** – Upserts into Supabase using the `(source, product_url)` unique constraint.
