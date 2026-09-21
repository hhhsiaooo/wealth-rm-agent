from __future__ import annotations

import hashlib
import os

import pytest

from rag.product_knowledge import PRODUCTS
from rag.retrieval import build_collection, get_chroma_client, search


class FakeEmbeddingClient:
    """Deterministic fake embeddings -- no network call, no API cost.
    Not semantically meaningful; only used to test the mechanical parts
    (indexing, metadata filtering, idempotency, result shape)."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dim]]


@pytest.fixture
def fake_client():
    return FakeEmbeddingClient()


# -- build_collection ---------------------------------------------------

def test_build_collection_indexes_all_30_products(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    assert collection.count() == 30


def test_build_collection_is_idempotent_on_rerun(tmp_path, fake_client):
    build_collection(fake_client, persist_path=tmp_path)
    build_collection(fake_client, persist_path=tmp_path)  # upsert, not add
    collection = get_chroma_client(tmp_path).get_or_create_collection("product_knowledge")
    assert collection.count() == 30


def test_build_collection_embeds_the_chunk_text_not_raw_metadata(tmp_path, fake_client):
    build_collection(fake_client, persist_path=tmp_path)
    assert len(fake_client.calls) == 30
    assert any("貨幣市場基金" in call for call in fake_client.calls)


# -- search: structure and filtering -----------------------------------------

def test_search_returns_requested_number_of_results(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    hits = search(collection, fake_client, "投資建議", n_results=5)
    assert len(hits) == 5
    for hit in hits:
        assert set(hit.keys()) == {"product_id", "document", "metadata", "distance"}


def test_search_filtered_by_category_only_returns_that_category(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    hits = search(collection, fake_client, "基金推薦", product_category="Fund", n_results=10)
    assert len(hits) == 10
    assert all(hit["metadata"]["product_category"] == "Fund" for hit in hits)


def test_search_filtered_by_td_category(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    hits = search(collection, fake_client, "定存", product_category="TD", n_results=10)
    assert len(hits) == 10
    assert all(hit["metadata"]["product_category"] == "TD" for hit in hits)


def test_search_filtered_by_fx_category(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    hits = search(collection, fake_client, "外幣", product_category="FX", n_results=10)
    assert len(hits) == 10
    assert all(hit["metadata"]["product_category"] == "FX" for hit in hits)


def test_search_without_category_can_return_multiple_categories(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    hits = search(collection, fake_client, "投資建議", n_results=30)
    categories = {hit["metadata"]["product_category"] for hit in hits}
    assert len(categories) > 1


# -- real API integration test (opt-in only) ---------------------------------

@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set -- skipping real Gemini embedding call",
)
def test_real_gemini_embedding_search_finds_a_relevant_fund(tmp_path):
    from rag.retrieval import GeminiEmbeddingClient

    client = GeminiEmbeddingClient()
    collection = build_collection(client, persist_path=tmp_path)
    hits = search(collection, client, "中高風險、適合積極投資、投資期間3年以上的基金產品", product_category="Fund", n_results=3)

    assert len(hits) == 3
    assert all(hit["metadata"]["product_category"] == "Fund" for hit in hits)
    assert any(hit["metadata"]["risk_level"] in ("Medium", "High") for hit in hits)
