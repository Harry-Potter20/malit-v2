from __future__ import annotations

import logging
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_phash(image_path: Path, hash_size: int = 16) -> imagehash.ImageHash:
    with Image.open(image_path).convert("RGB") as img:
        return imagehash.phash(img, hash_size=hash_size)


def deduplicate(
    image_paths: list[Path],
    hash_size: int = 16,
    hamming_threshold: int = 4,
) -> tuple[list[Path], list[Path]]:
    """
    Remove near-duplicate images using pHash with Hamming distance.

    Returns
    -------
    kept : list[Path]   — unique images to retain
    removed : list[Path] — duplicates detected and removed
    """
    hashes: dict[str, Path] = {}
    kept: list[Path] = []
    removed: list[Path] = []

    logger.info("Computing pHashes for %d images (hash_size=%d)…", len(image_paths), hash_size)

    for path in tqdm(image_paths, desc="Deduplication"):
        try:
            h = compute_phash(path, hash_size)
        except Exception as exc:
            logger.warning("Could not hash %s: %s — keeping it.", path, exc)
            kept.append(path)
            continue

        is_dup = False
        for existing_h_str, existing_path in hashes.items():
            existing_h = imagehash.hex_to_hash(existing_h_str)
            if (h - existing_h) <= hamming_threshold:
                removed.append(path)
                is_dup = True
                break

        if not is_dup:
            hashes[str(h)] = path
            kept.append(path)

    logger.info(
        "Deduplication complete: %d kept, %d removed (%.1f%% reduction)",
        len(kept),
        len(removed),
        100 * len(removed) / max(len(image_paths), 1),
    )
    return kept, removed
