"""Image and text embeddings using google/siglip-base-patch16-384.

This model produces 768-dimensional embeddings for both images and text
using SigLIP's dual-encoder architecture.

IMPORTANT: In transformers 5.x, SigLIP's get_text_features()/get_image_features()
helpers are broken. We access the sub-models directly:
  - text: text_model → pooler_output → text_model.head (Linear 768→768) → L2 normalize
  - image: vision_model → pooler_output (via SiglipMultiheadAttentionPoolingHead) → L2 normalize
"""

import logging
import io
from typing import List, Optional

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel
import numpy as np

from config import EMBEDDING_MODEL, BATCH_SIZE_EMBEDDINGS

logger = logging.getLogger(__name__)


class SiglipEmbedder:
    """Wrapper around SigLIP model for image and text embeddings."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading SigLIP model '%s' on %s ...", EMBEDDING_MODEL, self.device)
        self.model = SiglipModel.from_pretrained(EMBEDDING_MODEL).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL)
        logger.info("SigLIP model loaded successfully.")

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL and return as PIL Image."""
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            logger.warning("Failed to download image %s: %s", url, e)
            return None

    def embed_images(self, image_urls: List[str]) -> List[Optional[np.ndarray]]:
        """Compute 768-dim image embeddings for a list of image URLs.

        Uses vision_model → pooler_output (attention pooling head) → L2 normalize.

        Returns list of numpy arrays (1D, 768-dim) or None for failed downloads.
        """
        results: List[Optional[np.ndarray]] = [None] * len(image_urls)

        for batch_start in range(0, len(image_urls), BATCH_SIZE_EMBEDDINGS):
            batch_end = min(batch_start + BATCH_SIZE_EMBEDDINGS, len(image_urls))
            batch_urls = image_urls[batch_start:batch_end]

            # Download images
            images = []
            valid_indices = []
            for i, url in enumerate(batch_urls):
                img = self._download_image(url)
                if img is not None:
                    images.append(img)
                    valid_indices.append(batch_start + i)

            if not images:
                continue

            try:
                inputs = self.processor(
                    text=None, images=images, return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(self.device)

                with torch.no_grad():
                    vision_out = self.model.vision_model(pixel_values=pixel_values)
                    # pooler_output comes from SiglipMultiheadAttentionPoolingHead
                    # which produces the final projected 768-dim visual features
                    image_embeds = vision_out.pooler_output
                    # L2 normalize
                    image_embeds = image_embeds / image_embeds.norm(
                        p=2, dim=-1, keepdim=True,
                    )

                for idx, embed in zip(valid_indices, image_embeds.cpu().numpy()):
                    results[idx] = embed

                logger.debug(
                    "Embedded batch of %d images (%d-%d)",
                    len(images), batch_start, batch_end - 1,
                )

            except Exception as e:
                logger.warning(
                    "Failed to embed image batch %d-%d: %s",
                    batch_start, batch_end - 1, e,
                )

        return results

    def embed_text(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Compute 768-dim text embeddings for a list of text strings.

        Uses text_model → pooler_output → text_model.head (Linear 768→768) → L2 normalize.

        Returns list of numpy arrays (1D, 768-dim).
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)

        for batch_start in range(0, len(texts), BATCH_SIZE_EMBEDDINGS):
            batch_end = min(batch_start + BATCH_SIZE_EMBEDDINGS, len(texts))
            batch_texts = texts[batch_start:batch_end]

            try:
                inputs = self.processor(
                    text=batch_texts, images=None,
                    padding="max_length", truncation=True,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(self.device)

                with torch.no_grad():
                    text_out = self.model.text_model(input_ids=input_ids)
                    # SigLIP text model returns BaseModelOutputWithPooling
                    # pooler_output = EOS-token pooled representation [batch, 768]
                    pooled = text_out.pooler_output
                    # Apply text projection (head = Linear 768→768 in transformers 5.x)
                    text_embeds = self.model.text_model.head(pooled)
                    # L2 normalize
                    text_embeds = text_embeds / text_embeds.norm(
                        p=2, dim=-1, keepdim=True,
                    )

                for i, embed in enumerate(text_embeds.cpu().numpy()):
                    results[batch_start + i] = embed

                logger.debug(
                    "Embedded batch of %d texts (%d-%d)",
                    len(batch_texts), batch_start, batch_end - 1,
                )

            except Exception as e:
                logger.warning(
                    "Failed to embed text batch %d-%d: %s",
                    batch_start, batch_end - 1, e,
                )

        return results

    def embed_single_image(self, image_url: str) -> Optional[np.ndarray]:
        """Convenience: embed a single image URL."""
        results = self.embed_images([image_url])
        return results[0]

    def embed_single_text(self, text: str) -> Optional[np.ndarray]:
        """Convenience: embed a single text string."""
        results = self.embed_text([text])
        return results[0]
