from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _hash_one(path: Path, hash_size: int) -> tuple[Path, np.ndarray | None]:
    try:
        with Image.open(path).convert("RGB") as img:
            h = imagehash.phash(img, hash_size=hash_size)
        return path, np.packbits(h.hash.flatten())  # pack 256 bits → 32 uint8
    except Exception as exc:
        logger.warning("Could not hash %s: %s — keeping it.", path, exc)
        return path, None


def deduplicate(
    image_paths: list[Path],
    hash_size: int = 16,
    hamming_threshold: int = 4,
    num_workers: int | None = None,
) -> tuple[list[Path], list[Path]]:
    """
    Remove near-duplicate images using pHash + Hamming distance.

    Speed improvements over naive O(n²) loop:
      - Parallel image hashing via ThreadPoolExecutor (I/O + numpy release GIL)
      - Vectorized Hamming via packed bits: XOR + unpackbits (8× less memory)
    """
    n = len(image_paths)
    workers = num_workers or min(8, os.cpu_count() or 4)
    bits = hash_size * hash_size
    packed_len = bits // 8  # bytes per hash after np.packbits

    logger.info("Hashing %d images in parallel (workers=%d)…", n, workers)

    # ── Step 1: Parallel hash computation ────────────────────────────────────
    hash_map: dict[Path, np.ndarray | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_hash_one, p, hash_size): p for p in image_paths}
        for fut in tqdm(as_completed(futures), total=n, desc="Hashing", unit="img"):
            path, h = fut.result()
            hash_map[path] = h

    # ── Step 2: Vectorized deduplication ─────────────────────────────────────
    # Pre-allocate: (n, packed_len) uint8  →  e.g. 55K × 32 = 1.7 MB
    stored = np.empty((n, packed_len), dtype=np.uint8)
    m = 0
    kept: list[Path] = []
    removed: list[Path] = []

    for path in tqdm(image_paths, desc="Deduplication", unit="img"):
        h = hash_map.get(path)
        if h is None:  # hashing failed — keep the image
            kept.append(path)
            continue

        is_dup = False
        if m > 0:
            # XOR packed bytes, unpack to bits, sum per row → Hamming distances
            xor = stored[:m] ^ h                          # (m, packed_len) uint8
            dists = np.unpackbits(xor, axis=1).sum(axis=1)  # (m,) int
            is_dup = bool(dists.min() <= hamming_threshold)

        if is_dup:
            removed.append(path)
        else:
            stored[m] = h
            m += 1
            kept.append(path)

    logger.info(
        "Deduplication complete: %d kept, %d removed (%.1f%% reduction)",
        len(kept), len(removed), 100 * len(removed) / max(n, 1),
    )
    return kept, removed
