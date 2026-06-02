"""Configuration for the Menyard fashion scraper.

Supports environment variable overrides for CI:
- SUPABASE_URL / SUPABASE_KEY: Supabase credentials
"""

import os

# Shopify store info
SHOP_DOMAIN = "menyardhomme.com"
SHOP_URL = f"https://www.{SHOP_DOMAIN}"

# Collections to scrape with their base categories
COLLECTIONS = [
    {
        "handle": "tops-all",
        "label": "TOPS ALL",
        "categories": ["Tops"],
    },
    {
        "handle": "bottoms-all",
        "label": "BOTTOMS ALL",
        "categories": ["Bottoms"],
    },
    {
        "handle": "all-accessoires",
        "label": "ALL ACCESSOIRES",
        "categories": ["Accessories"],
    },
    {
        "handle": "sale",
        "label": "SALE",
        "categories": None,  # Determined dynamically from product's other collections
    },
]

# Supabase — values can be overridden via environment variables for CI
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "https://yqawmzggcgpeyaaynrjk.supabase.co",
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
SUPABASE_TABLE = "products"

# Embeddings
EMBEDDING_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
BATCH_SIZE_EMBEDDINGS = 16  # Number of images/texts to batch through the model

# Static product fields
SOURCE = "scraper-menyard"
BRAND = "Menyard"
GENDER = "men"
SECOND_HAND = False
COUNTRY = "NL"

# Request settings
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
SHOPIFY_PAGE_LIMIT = 250

# Currency (base currency from Shopify API is EUR)
# If other currencies are needed, we'd need an external conversion API
BASE_CURRENCY = "EUR"
