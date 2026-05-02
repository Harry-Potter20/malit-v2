from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


class EmbeddingExtractor:
    """Extracts 512-d embeddings from MALITV2 after global average pooling."""

    def __init__(self, model: nn.Module, device: str | torch.device | None = None):
        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        self.model.to(self.device)
        all_embs, all_labels = [], []

        for imgs, labels in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(self.device)
            # Hook into GAP output by running forward and capturing pooled features
            embs = self._forward_embedding(imgs)
            all_embs.append(embs.cpu().numpy())
            all_labels.extend(labels.tolist())

        return np.concatenate(all_embs, axis=0), np.array(all_labels)

    def _forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vector (before classifier)."""
        # Run model forward; extract features before the classifier head
        # MALITV2 exposes this via extract_embedding()
        if hasattr(self.model, "extract_embedding"):
            return self.model.extract_embedding(x)

        # Fallback: hook-based extraction
        features = {}

        def hook(module, inp, out):
            features["gap"] = out

        handle = self.model.gap.register_forward_hook(hook)
        self.model(x)
        handle.remove()
        return features["gap"].flatten(1)


class CaseBasedReasoning:
    """
    FAISS-based nearest-neighbor retrieval for clinical interpretability.

    Answers: "Has the model seen similar cases before, and what were their labels?"
    """

    def __init__(self, k: int = 5, use_faiss: bool = True):
        self.k = k
        self.faiss = _try_import_faiss() if use_faiss else None
        self.index = None
        self.train_labels: np.ndarray | None = None
        self.train_paths: list[str] | None = None
        self._dim: int | None = None

    def build_index(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        paths: list[str] | None = None,
    ) -> None:
        embeddings = embeddings.astype(np.float32)
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        self._dim = embeddings.shape[1]
        self.train_labels = labels
        self.train_paths = paths or [f"sample_{i}" for i in range(len(labels))]

        if self.faiss is not None:
            self.index = self.faiss.IndexFlatIP(self._dim)  # Inner product = cosine after L2 norm
            self.index.add(embeddings)
            logger.info("FAISS index built: %d vectors, dim=%d", len(embeddings), self._dim)
        else:
            logger.warning("faiss-cpu not available — using sklearn NearestNeighbors fallback.")
            from sklearn.neighbors import NearestNeighbors
            self.index = NearestNeighbors(n_neighbors=self.k, metric="cosine")
            self.index.fit(embeddings)
            self._faiss_fallback = True

    def retrieve(
        self, query_embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (indices, distances) each shape (n_queries, k).
        """
        query_embeddings = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / (norms + 1e-8)

        if self.faiss is not None and isinstance(self.index, self.faiss.Index):
            scores, indices = self.index.search(query_embeddings, self.k)
            # FAISS inner product — higher is closer; convert to distance
            distances = 1.0 - scores
        else:
            distances_raw, indices = self.index.kneighbors(query_embeddings)
            distances = distances_raw

        return indices, distances

    def consistency_score(self, indices: np.ndarray, query_label: int | None = None) -> float:
        """Fraction of k-neighbors sharing the same label as the majority."""
        neighbor_labels = self.train_labels[indices]
        majority = np.bincount(neighbor_labels).argmax()
        if query_label is not None:
            return float((neighbor_labels == query_label).mean())
        return float((neighbor_labels == majority).mean())

    def run_on_embeddings(
        self,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
    ) -> dict[str, Any]:
        assert self.index is not None, "Call build_index() first."
        indices, distances = self.retrieve(query_embeddings)

        results = []
        all_consistency = []
        for i, (idx_row, dist_row, q_label) in enumerate(
            zip(indices, distances, query_labels)
        ):
            neighbor_labels = self.train_labels[idx_row].tolist()
            neighbor_paths = [self.train_paths[j] for j in idx_row]
            consistency = self.consistency_score(idx_row, int(q_label))
            all_consistency.append(consistency)
            results.append({
                "query_idx": i,
                "query_label": int(q_label),
                "neighbor_indices": idx_row.tolist(),
                "neighbor_labels": neighbor_labels,
                "neighbor_paths": neighbor_paths,
                "distances": dist_row.tolist(),
                "consistency_score": consistency,
            })

        consistency_arr = np.array(all_consistency)
        errors = (query_embeddings.shape[0] and
                  (query_labels != np.array([np.bincount(self.train_labels[r]).argmax()
                                             for r in indices]))).astype(float)

        # Failure check
        if distances.std() < 1e-6:
            logger.error("FAILURE: All CBR distances equal — possible normalization issue.")

        return {
            "neighbors": results,
            "mean_consistency": float(consistency_arr.mean()),
            "std_consistency": float(consistency_arr.std()),
        }
