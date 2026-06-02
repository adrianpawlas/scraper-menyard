"""Image URL filtering logic for Menyard product images.

Rules to identify "good" product images (product on blank background):
- Filename does NOT contain: cropped, e-commerce/ecommerce, pro-capture/capture_one
- Filename does NOT start with 4 digits followed by underscore (batch exports)
- Filename does NOT contain date patterns like 29-08-2025
- Non-ALL-CAPS filenames with 3+ underscore segments are filtered (descriptive names)
- Single underscore + known color word at end is filtered (model/lifestyle shots)
"""

import re
from urllib.parse import urlparse, unquote


# Patterns that identify "bad" images (lifestyle, cropped, detail views, etc.)
BAD_PATTERNS = [
    re.compile(r'cropped', re.IGNORECASE),
    re.compile(r'e[\s-]?commerce', re.IGNORECASE),
    re.compile(r'pro[\s-]?capture', re.IGNORECASE),
    re.compile(r'capture[\s-]?one', re.IGNORECASE),
    re.compile(r'^\d{4}_'),  # Starts with 4 digits + underscore (0022_...)
    re.compile(r'\d{2}-\d{2}-\d{4}'),  # Date pattern like 29-08-2025
]

# Common color words that indicate a filename is descriptive (lifestyle shot)
# rather than a clean product shot.
COLOR_WORDS = {
    "black", "white", "grey", "gray", "blue", "red", "green", "yellow", "navy",
    "brown", "pink", "purple", "orange", "beige", "cream", "khaki", "olive",
    "tan", "teal", "burgundy", "charcoal", "ivory", "mauve", "indigo", "violet",
    "coral", "maroon", "gold", "silver", "bronze", "copper", "rust", "blush",
    "nude", "camel", "chocolate", "mint", "sage", "forest", "army", "wine",
    "plum", "lavender", "lilac", "peach", "salmon", "mustard", "cognac", "denim",
    "offwhite", "off-white", "ecru", "taupe", "terracotta", "clay", "sand",
    "smoke", "steel", "midnight", "royal", "sky", "ocean", "emerald",
    "jade", "ruby", "rose", "dusty", "faded", "washed", "raw", "acid",
}


def _extract_filename(url: str) -> str:
    """Extract the filename stem (without extension) from an image URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    filename = path.rsplit("/", 1)[-1] if "/" in path else path
    # Remove the file extension
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return stem


def is_good_product_image(image_url: str) -> bool:
    """Determine if an image URL is a 'good' product shot vs a lifestyle/bad shot."""
    stem = _extract_filename(image_url)
    if not stem:
        return False

    # 1. Check specific bad patterns (cropped, e-commerce, date, batch exports, etc.)
    for pattern in BAD_PATTERNS:
        if pattern.search(stem):
            return False

    segments = stem.split("_")

    # 2. For 3+ underscore segments: allow ALL-CAPS stems (Shopify naming convention),
    #    but filter non-ALL-CAPS descriptive names like "Tote_bag_coffee_and_culture_brown"
    #    or "Regatta_Black_Melee"
    if len(segments) >= 3 and not stem.isupper():
        return False

    # 3. Single underscore where the LAST segment is a known color word
    #    indicates a lifestyle/model labeling pattern like "Regatta_Grey"
    if len(segments) == 2 and segments[-1].lower() in COLOR_WORDS:
        return False

    return True


def filter_product_images(
    images: list[dict],
    max_additional: int = 10,
) -> tuple[str | None, list[str]]:
    """Filter product images and return (main_image_url, additional_image_urls).

    Args:
        images: List of image dicts from Shopify product JSON. Each must have
                'src' and 'position' keys.
        max_additional: Max number of additional images to include.

    Returns:
        Tuple of (main_image_url or None, list of additional image URLs).
    """
    # Sort by position
    sorted_images = sorted(images, key=lambda img: img.get("position", 999))

    # Filter to only good images
    good_images = [img for img in sorted_images if is_good_product_image(img["src"])]

    if not good_images:
        # Fallback: use first sorted image even if it's "bad"
        # to ensure we always have at least one image
        if sorted_images:
            good_images = [sorted_images[0]]
        else:
            return None, []

    main_image = good_images[0]["src"]
    additional = [img["src"] for img in good_images[1:][:max_additional]]

    return main_image, additional
