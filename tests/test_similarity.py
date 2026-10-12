"""Tests for sentence similarity using Model2Vec static embeddings."""

import numpy as np
import pytest
from gemini_score.similarity import DEFAULT_SIMILARITY_MODEL, SentenceSimilarity


@pytest.fixture(scope="module")
def similarity_engine():
    return SentenceSimilarity.get_instance(DEFAULT_SIMILARITY_MODEL)


def test_similarity_initialization(similarity_engine):
    assert similarity_engine.dim > 0
    assert similarity_engine.model_name == DEFAULT_SIMILARITY_MODEL


def test_encode_empty(similarity_engine):
    empty = similarity_engine.encode([])
    assert empty.shape == (0, similarity_engine.dim)


def test_encode_strings(similarity_engine):
    texts = ["Hello world", "Good morning", "Testing static embeddings"]
    embs = similarity_engine.encode(texts, normalize=True)
    assert embs.shape == (3, similarity_engine.dim)
    # Check L2 norm is ~1.0
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-4)


def test_encode_large_batch(similarity_engine):
    texts = [f"This is sentence number {i}" for i in range(300)]
    embs = similarity_engine.encode(texts, batch_size=64, normalize=True)
    assert embs.shape == (300, similarity_engine.dim)


def test_compute_similarity_identical(similarity_engine):
    texts = ["The cat sat on the mat.", "The sun is shining bright."]
    scores = similarity_engine.compute_similarity(texts, texts, scale="percent")
    assert len(scores) == 2
    # Identical texts should have ~100% similarity
    assert scores[0] >= 99.0
    assert scores[1] >= 99.0


def test_compute_similarity_semantic_comparison(similarity_engine):
    translations = ["The puppy barked loudly.", "He is writing Python code."]
    references = ["A small dog barked loud.", "The weather is very cold today."]
    scores = similarity_engine.compute_similarity(translations, references, scale="percent")
    # Puppy barking vs dog barking should be substantially higher than Python vs weather
    assert scores[0] > scores[1]
    assert scores[0] > 50.0


def test_compute_similarity_scales(similarity_engine):
    t = ["Good morning"]
    r = ["Good morning"]

    pct = similarity_engine.compute_similarity(t, r, scale="percent")
    unit = similarity_engine.compute_similarity(t, r, scale="unit")
    raw = similarity_engine.compute_similarity(t, r, scale="raw")

    assert 99.0 <= pct[0] <= 100.01
    assert 0.99 <= unit[0] <= 1.01
    assert 0.99 <= raw[0] <= 1.01

    with pytest.raises(ValueError):
        similarity_engine.compute_similarity(t, r, scale="invalid_scale")


def test_compute_similarity_multi_reference(similarity_engine):
    translations = ["The car stopped at the traffic light."]
    # Two references: one unrelated, one very close
    references = [["A bird flew over the sky.", "The automobile halted at the red signal."]]
    scores = similarity_engine.compute_similarity(translations, references, scale="percent")
    assert len(scores) == 1
    # Multi-reference picks the maximum similarity
    assert scores[0] > 60.0


def test_compute_similarity_mismatched_lengths(similarity_engine):
    with pytest.raises(ValueError, match="Mismatch"):
        similarity_engine.compute_similarity(["A", "B"], ["A"])
