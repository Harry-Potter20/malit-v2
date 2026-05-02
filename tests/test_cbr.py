"""Tests for Case-Based Reasoning module."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestCaseBasedReasoning:
    @pytest.fixture
    def embeddings_and_labels(self):
        np.random.seed(42)
        # 100 train, 20 test — 2 classes
        train_embs = np.random.randn(100, 512).astype(np.float32)
        train_labels = np.array([i % 2 for i in range(100)])
        test_embs = np.random.randn(20, 512).astype(np.float32)
        test_labels = np.array([i % 2 for i in range(20)])
        return train_embs, train_labels, test_embs, test_labels

    def _build_cbr(self, train_embs, train_labels):
        from src.explainability.case_based_reasoning import CaseBasedReasoning
        cbr = CaseBasedReasoning(k=5, use_faiss=False)  # sklearn fallback
        cbr.build_index(train_embs, train_labels)
        return cbr

    def test_build_index(self, embeddings_and_labels):
        train_embs, train_labels, _, _ = embeddings_and_labels
        cbr = self._build_cbr(train_embs, train_labels)
        assert cbr.index is not None
        assert cbr.train_labels is not None

    def test_retrieve_returns_k_neighbors(self, embeddings_and_labels):
        train_embs, train_labels, test_embs, test_labels = embeddings_and_labels
        cbr = self._build_cbr(train_embs, train_labels)
        indices, distances = cbr.retrieve(test_embs[:5])
        assert indices.shape == (5, 5)
        assert distances.shape == (5, 5)

    def test_indices_in_valid_range(self, embeddings_and_labels):
        train_embs, train_labels, test_embs, _ = embeddings_and_labels
        cbr = self._build_cbr(train_embs, train_labels)
        indices, _ = cbr.retrieve(test_embs[:5])
        assert (indices >= 0).all()
        assert (indices < len(train_labels)).all()

    def test_distances_non_negative(self, embeddings_and_labels):
        train_embs, train_labels, test_embs, _ = embeddings_and_labels
        cbr = self._build_cbr(train_embs, train_labels)
        _, distances = cbr.retrieve(test_embs[:5])
        assert (distances >= 0).all()

    def test_consistency_score_range(self, embeddings_and_labels):
        train_embs, train_labels, test_embs, test_labels = embeddings_and_labels
        cbr = self._build_cbr(train_embs, train_labels)
        indices, _ = cbr.retrieve(test_embs[:5])
        for row in indices:
            score = cbr.consistency_score(row)
            assert 0.0 <= score <= 1.0

    def test_perfect_neighbors_consistency_1(self):
        """If all neighbors have label 1, consistency with query=1 should be 1."""
        from src.explainability.case_based_reasoning import CaseBasedReasoning
        train_embs = np.random.randn(50, 32).astype(np.float32)
        train_labels = np.ones(50, dtype=int)
        cbr = CaseBasedReasoning(k=5, use_faiss=False)
        cbr.build_index(train_embs, train_labels)
        indices, _ = cbr.retrieve(train_embs[:1])
        score = cbr.consistency_score(indices[0], query_label=1)
        assert score == pytest.approx(1.0)

    def test_run_on_embeddings_structure(self, embeddings_and_labels):
        train_embs, train_labels, test_embs, test_labels = embeddings_and_labels
        cbr = self._build_cbr(train_embs, train_labels)
        results = cbr.run_on_embeddings(test_embs, test_labels)
        assert "neighbors" in results
        assert "mean_consistency" in results
        assert 0.0 <= results["mean_consistency"] <= 1.0

    def test_neighbor_count(self, embeddings_and_labels):
        train_embs, train_labels, test_embs, test_labels = embeddings_and_labels
        cbr = self._build_cbr(train_embs, train_labels)
        results = cbr.run_on_embeddings(test_embs, test_labels)
        for neighbor in results["neighbors"]:
            assert len(neighbor["neighbor_indices"]) == 5

    def test_embedding_extractor_shape(self, tiny_model, sample_loader):
        from src.explainability.case_based_reasoning import EmbeddingExtractor
        extractor = EmbeddingExtractor(tiny_model)
        embs, labels = extractor.extract(sample_loader)
        assert embs.shape[1] == 512
        assert len(labels) == 8
