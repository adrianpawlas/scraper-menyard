"""Image URL filtering logic for Menyard product images.

Uses a scoring-based approach to identify the best product images:
1. Lifestyle/season keywords (DROP, DETAIL, FULL_FIT, SS26, cropped, etc.) are filtered out
2. Images with UUIDs (Shopify variant identifiers) are prioritized
3. Numbered suffixes (like _2, _4, _5) indicate secondary/model shots
4. Color variant words from the product title are used to match variant-specific images
5. Images are scored and ranked; the highest-scored becomes the main image
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, unquote


# Lifestyle/keyword patterns that identify "bad" images
LIFESTYLE_PATTERNS = [
    re.compile(r'drop', re.IGNORECASE),  # Collection lookbook drops
    re.compile(r'detail', re.IGNORECASE),  # Detail shots
    re.compile(r'full[_-]?fit', re.IGNORECASE),  # Full outfit model shots
    re.compile(r'ss\d{2}', re.IGNORECASE),  # Season codes like SS26
    re.compile(r'sbl', re.IGNORECASE),  # Model naming pattern (SBL_-_Jasmijn)
    re.compile(r'img_', re.IGNORECASE),  # Camera-generated filenames
    re.compile(r'cropped', re.IGNORECASE),  # Cropped images
    re.compile(r'generative[_-]?fill', re.IGNORECASE),  # AI-generated backgrounds
    re.compile(r'look[_-]?book', re.IGNORECASE),  # Lookbook shots
    re.compile(r'e[\s-]?commerce', re.IGNORECASE),
    re.compile(r'pro[\s-]?capture', re.IGNORECASE),
    re.compile(r'capture[\s-]?one', re.IGNORECASE),
    re.compile(r'\d{2}-\d{2}-\d{4}'),  # Date pattern like 29-08-2025
]

# UUID pattern: 8-4-4-4-12 hex digits (Shopify variant image identifier)
UUID_PATTERN = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.IGNORECASE,
)

# Numbered suffix: underscore followed by digits at end (but not _V# like _V2)
NUMBERED_SUFFIX = re.compile(r'_(?![Vv])\d+$')


def _extract_filename(url: str) -> str:
    """Extract the filename stem (without extension) from an image URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    filename = path.rsplit("/", 1)[-1] if "/" in path else path
    # Remove the file extension
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return stem


def _has_lifestyle_keyword(stem: str) -> bool:
    """Check if the filename stem contains lifestyle/season keywords."""
    for pattern in LIFESTYLE_PATTERNS:
        if pattern.search(stem):
            return True
    return False


def _has_uuid(stem: str) -> bool:
    """Check if filename contains a Shopify variant UUID."""
    return bool(UUID_PATTERN.search(stem))


def _has_numbered_suffix(stem: str) -> bool:
    """Check if filename ends with _NUMBER (but not _VERSION like _V2)."""
    return bool(NUMBERED_SUFFIX.search(stem))


def _extract_variant_words(product_title: str | None) -> set[str]:
    """Extract color variant words from product title.

    Product titles in Menyard follow the format:
        "Product Name | Color Variant"
    e.g., "Luberon Shirt | Mid Blue/White" → {"MID", "BLUE", "WHITE"}
    """
    if not product_title:
        return set()
    # Extract part after "|" if present (the color variant)
    if "|" in product_title:
        variant_part = product_title.split("|", 1)[1].strip()
    else:
        variant_part = product_title
    # Split by /, whitespace, or comma, then collect individual words
    words: set[str] = set()
    for part in re.split(r'[/\s,]+', variant_part):
        word = part.strip().upper()
        if word and len(word) > 1:  # Skip single characters
            words.add(word)
    return words


def score_product_image(stem: str, variant_words: set[str]) -> int:
    """Score an image filename stem. Higher = better product image.

    Scoring tiers:
        100   → Has UUID (Shopify variant-specific product image)
        70-99 → Good variant-matched product image
        50-69 → Acceptable product image (base score)
        10-49 → Low quality (numbered suffix, poor variant match)
        -100  → Lifestyle/model/detail shot (filtered out)
    """
    stem_upper = stem.upper()

    # Lifestyle keywords → immediately bad
    if _has_lifestyle_keyword(stem_upper):
        return -100

    # UUID present → variant-specific product image (best quality)
    if _has_uuid(stem):
        return 100

    score = 50  # base score for a passable image

    # Penalty for numbered suffix (secondary shots like _2, _4, _5)
    if _has_numbered_suffix(stem):
        score -= 40

    # Bonus for matching color variant words from product title
    if variant_words:
        matches = sum(1 for w in variant_words if w in stem_upper)
        match_ratio = matches / len(variant_words)
        score += int(match_ratio * 40)  # up to +40 for perfect match
        if match_ratio < 0.5:
            score -= 20  # penalize if mostly unrelated to variant

    # Small bonus for ALL-CAPS (Shopify naming convention)
    if stem.isupper():
        score += 5

    return score


def filter_product_images(
    images: list[dict],
    max_additional: int = 10,
    product_title: str | None = None,
) -> tuple[str | None, list[str]]:
    """Filter product images and return (main_image_url, additional_image_urls).

    Uses a scoring system:
    1. Images with lifestyle keywords are filtered out
    2. Remaining images are scored based on UUID presence, color variant
       matching, numbered suffix, and naming conventions
    3. The highest-scored image (by position tiebreaker) becomes the main image
    4. Additional images above a quality threshold are included

    Args:
        images: List of image dicts from Shopify product JSON.
                Each must have 'src' and 'position' keys.
        max_additional: Max number of additional images to include.
        product_title: Product title used to extract color variant words.
                       Pass None if unavailable.

    Returns:
        Tuple of (main_image_url or None, list of additional image URLs).
    """
    if not images:
        return None, []

    variant_words = _extract_variant_words(product_title)

    # Score all images
    scored_images: list[tuple[int, int, str]] = []
    for img in images:
        src = img.get("src", "")
        if not src:
            continue
        stem = _extract_filename(src)
        score = score_product_image(stem, variant_words)
        position = img.get("position", 999)
        scored_images.append((score, position, src))

    if not scored_images:
        return None, []

    # Fallback: if ALL images are lifestyle (all negative), use the first by position
    all_lifestyle = all(score < 0 for score, _, _ in scored_images)
    if all_lifestyle:
        scored_images.sort(key=lambda x: x[1])
        main_image = scored_images[0][2]
        return main_image, []

    # Sort by score (descending), then position (ascending) for tiebreaking
    scored_images.sort(key=lambda x: (-x[0], x[1]))

    main_image = scored_images[0][2]

    # Additional images: include those with score >= 40 (reasonable quality)
    additional = [
        src for score, pos, src in scored_images[1:]
        if score >= 40
    ][:max_additional]

    return main_image, additional
