"""Tests for GeminiEmbeddings and Model2VecSimilarity engines."""

import numpy as np
import pytest
from gemini_score.similarity import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_MODEL,
    GeminiEmbeddings,
    Model2VecSimilarity,
    _pseudo_embed,
)


def make_mock_embedding_client():
    def mock_embed(texts, is_embed=True):
        return [_pseudo_embed(t) for t in texts]

    return mock_embed


@pytest.fixture(scope="module")
def gemini_engine():
    return GeminiEmbeddings(client=make_mock_embedding_client(), model="gemini-embedding-001")


@pytest.fixture(scope="module")
def model2vec_engine():
    return Model2VecSimilarity(DEFAULT_SIMILARITY_MODEL)


# --- Tests for GeminiEmbeddings (Default) ---


def test_gemini_embeddings_encode_empty(gemini_engine):
    empty = gemini_engine.encode([])
    assert empty.shape == (0, 0)


def test_gemini_embeddings_encode_strings(gemini_engine):
    texts = ["Hello world", "Good morning", "Testing gemini embeddings"]
    embs = gemini_engine.encode(texts, normalize=True)
    assert embs.shape == (3, 64)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-4)


def test_gemini_embeddings_caching():
    calls = 0

    def mock_client(texts, is_embed=True):
        nonlocal calls
        calls += 1
        return [_pseudo_embed(t) for t in texts]

    engine = GeminiEmbeddings(client=mock_client, enable_cache=True)
    engine.encode(["Repeated text"])
    engine.encode(["Repeated text"])
    assert calls == 1


def test_gemini_embeddings_compute_similarity_identical(gemini_engine):
    texts = ["The cat sat on the mat.", "The sun is shining bright."]
    scores = gemini_engine.compute_similarity(texts, texts, scale="percent")
    assert len(scores) == 2
    assert scores[0] >= 99.0
    assert scores[1] >= 99.0


def test_gemini_embeddings_scales(gemini_engine):
    t = ["Good morning"]
    r = ["Good morning"]

    pct = gemini_engine.compute_similarity(t, r, scale="percent")
    unit = gemini_engine.compute_similarity(t, r, scale="unit")
    raw = gemini_engine.compute_similarity(t, r, scale="raw")

    assert 99.0 <= pct[0] <= 100.01
    assert 0.99 <= unit[0] <= 1.01
    assert 0.99 <= raw[0] <= 1.01

    with pytest.raises(ValueError):
        gemini_engine.compute_similarity(t, r, scale="invalid_scale")


def test_gemini_embeddings_multi_reference(gemini_engine):
    translations = ["The car stopped at the traffic light."]
    references = [["A bird flew over the sky.", "The car stopped at the traffic light."]]
    scores = gemini_engine.compute_similarity(translations, references, scale="percent")
    assert len(scores) == 1
    assert scores[0] >= 99.0


def test_gemini_embeddings_mismatched_lengths(gemini_engine):
    with pytest.raises(ValueError, match="Mismatch"):
        gemini_engine.compute_similarity(["A", "B"], ["A"])


# --- Tests for Model2VecSimilarity (Optional Fallback) ---


def test_model2vec_initialization(model2vec_engine):
    assert model2vec_engine.dim > 0
    assert model2vec_engine.model_name == DEFAULT_SIMILARITY_MODEL


def test_model2vec_encode_and_similarity(model2vec_engine):
    t = ["The cat sat on the mat."]
    r = ["The cat sat on the mat."]
    scores = model2vec_engine.compute_similarity(t, r, scale="percent")
    assert scores[0] >= 99.0
