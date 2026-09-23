from __future__ import annotations

import hashlib
import os

import pytest

from rag.product_knowledge import PRODUCTS
from rag.retrieval import build_collection, get_chroma_client, load_or_build_collection, make_retriever, search


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


# -- embedding model precedence (mirrors GeminiLLMClient's GEMINI_MODEL) -----
# Uses a dummy in place of the real genai.Client so this never constructs a
# real client (no network call, and avoids combining a real genai.Client with
# chromadb in the same process -- known to segfault in the shared env when
# the client actually does work; see docs/ai_product_design.md 附錄).

class _DummyGenaiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key


@pytest.fixture
def dummy_genai_client(monkeypatch):
    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", _DummyGenaiClient)


def test_embedding_client_defaults_when_env_not_set(monkeypatch, dummy_genai_client):
    from rag.retrieval import DEFAULT_EMBEDDING_MODEL, GeminiEmbeddingClient

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert GeminiEmbeddingClient(api_key="x").model == DEFAULT_EMBEDDING_MODEL


def test_embedding_client_reads_env_model(monkeypatch, dummy_genai_client):
    from rag.retrieval import GeminiEmbeddingClient

    monkeypatch.setenv("EMBEDDING_MODEL", "env-embedding-model")
    assert GeminiEmbeddingClient(api_key="x").model == "env-embedding-model"


def test_embedding_client_explicit_model_wins_over_env(monkeypatch, dummy_genai_client):
    from rag.retrieval import GeminiEmbeddingClient

    monkeypatch.setenv("EMBEDDING_MODEL", "env-embedding-model")
    assert GeminiEmbeddingClient(model="explicit-model", api_key="x").model == "explicit-model"


# -- risk-level pre-filter (product risk must be within allowed set) ---------

def test_search_risk_filter_only_returns_allowed_risk_levels(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    hits = search(collection, fake_client, "基金", product_category="Fund",
                  n_results=30, risk_levels=["Low", "Medium"])
    assert len(hits) > 0
    assert {h["metadata"]["risk_level"] for h in hits} <= {"Low", "Medium"}
    assert all(h["metadata"]["product_category"] == "Fund" for h in hits)


def test_search_low_risk_filter_excludes_all_medium_and_high(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    hits = search(collection, fake_client, "定存", product_category="TD", n_results=30, risk_levels=["Low"])
    assert {h["metadata"]["risk_level"] for h in hits} == {"Low"}
    # TD_009 is Medium risk and must not appear for a Low-risk customer
    assert "TD_009" not in {h["product_id"] for h in hits}


def test_make_retriever_binds_filters(tmp_path, fake_client):
    collection = build_collection(fake_client, persist_path=tmp_path)
    retriever = make_retriever(collection, fake_client, n_results=5)
    hits = retriever("外幣", "FX", ["Low"])
    expected_ids = {
        x.metadata.product_id for x in PRODUCTS
        if x.metadata.product_category == "FX" and x.metadata.risk_level == "Low"
    }
    assert {h["product_id"] for h in hits} == expected_ids  # n_results=5 >= candidates, so all of them
    assert all(h["metadata"]["product_category"] == "FX" and h["metadata"]["risk_level"] == "Low" for h in hits)


def test_load_or_build_reuses_existing_collection_without_reembedding(tmp_path, fake_client):
    build_collection(fake_client, persist_path=tmp_path)
    calls_after_build = len(fake_client.calls)
    collection = load_or_build_collection(fake_client, persist_path=tmp_path)
    assert collection.count() == 30
    assert len(fake_client.calls) == calls_after_build  # no new embedding calls


def test_load_or_build_builds_when_missing(tmp_path, fake_client):
    collection = load_or_build_collection(fake_client, persist_path=tmp_path)
    assert collection.count() == 30
