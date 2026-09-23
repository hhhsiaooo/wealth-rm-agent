"""Phase 3.8 (WHAT): Retrieval.

Lightweight, free vector store: ChromaDB in embedded/local mode (no server),
persisted to a local folder. Embeddings are computed explicitly via the
Gemini Embedding API and passed into Chroma -- this bypasses Chroma's default
embedding function so we don't pull in sentence-transformers/torch, keeping
the dependency footprint small.

EmbeddingClient follows the same Dependency Injection pattern already used
for LLMClient in insight/insight_generator.py: swappable, and tests use a
deterministic fake instead of a real (paid/rate-limited) API call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from insight.insight_generator import load_env_file, resolve_model  # noqa: E402
from rag.product_knowledge import PRODUCTS, ProductKnowledge  # noqa: E402

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
CHROMA_PATH = Path(__file__).resolve(
).parents[2] / "data" / "synthetic" / "product_knowledge" / "chroma_db"
COLLECTION_NAME = "product_knowledge"


class EmbeddingClient(Protocol):
    def __call__(self, text: str) -> list[float]: ...


class GeminiEmbeddingClient:
    """Real embedding client backed by the Gemini Embedding API.

    Model precedence: explicit `model` argument > `.env`'s EMBEDDING_MODEL >
    DEFAULT_EMBEDDING_MODEL -- same pattern as GeminiLLMClient / GEMINI_MODEL
    (insight/insight_generator.py), so embedding model swaps for latency /
    quality experiments (Phase 5) don't need a code change either.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from google import genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=resolved_key)
        self._model = resolve_model(model, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    @property
    def model(self) -> str:
        return self._model

    def __call__(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self._model, contents=text)
        return list(result.embeddings[0].values)


def get_chroma_client(persist_path: Path = CHROMA_PATH):
    import chromadb

    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def build_collection(
    embedding_client: EmbeddingClient,
    persist_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    products: list[ProductKnowledge] | None = None,
):
    """Embed every product's chunk text and upsert into the Chroma collection.
    upsert (not add) so re-running this is idempotent."""
    products = products if products is not None else PRODUCTS
    client = get_chroma_client(persist_path)
    collection = client.get_or_create_collection(name=collection_name)

    ids = [p.metadata.product_id for p in products]
    documents = [p.to_chunk_text() for p in products]
    embeddings = [embedding_client(doc) for doc in documents]
    metadatas = [p.to_flat_metadata() for p in products]

    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=documents, metadatas=metadatas)
    return collection


# 這裡不呼叫會「寫文字」的 LLM，是用 Embedding 模型把 query 轉成向量，
# 跟資料庫裡每個產品的向量算相似度，回傳最接近的幾筆——本質是比對／
# 排序，不是生成內容。
def search(
    collection,
    embedding_client: EmbeddingClient,
    query: str,
    product_category: str | None = None,
    n_results: int = 3,
    risk_levels: Sequence[str] | None = None,
) -> list[dict]:
    """Embed the query, retrieve top-K, with optional metadata pre-filters
    (3.9 Filtering): product_category (exact) and risk_levels (product
    risk_level must be one of them). Filters are applied inside the vector
    store, so the top-K are all eligible -- not filtered after the fact.
    Returns a list of {product_id, document, metadata, distance} dicts,
    best match first."""
    query_embedding = embedding_client(query)
    conditions = []
    if product_category:
        conditions.append({"product_category": product_category})
    if risk_levels:
        conditions.append({"risk_level": {"$in": list(risk_levels)}})
    where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )

    hits = []
    for i in range(len(result["ids"][0])):
        hits.append({
            "product_id": result["ids"][0][i],
            "document": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i],
        })
    return hits


# (query, product_category, allowed product risk levels or None) -> hits
ProductRetriever = Callable[[str, str, Optional[Sequence[str]]], list[dict]]


def make_retriever(collection, embedding_client: EmbeddingClient, n_results: int = 3) -> ProductRetriever:
    """Bind a Chroma collection + embedding client into a plain function, so
    callers (nba_generator, agent tools) depend on a tiny interface and tests
    can inject a fake."""

    def retriever(query: str, product_category: str, risk_levels: Optional[Sequence[str]] = None) -> list[dict]:
        return search(
            collection, embedding_client, query,
            product_category=product_category, n_results=n_results, risk_levels=risk_levels,
        )

    return retriever


def load_or_build_collection(
    embedding_client: EmbeddingClient,
    persist_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    products: list[ProductKnowledge] | None = None,
):
    """Reuse an already-indexed collection; only call the (paid/rate-limited)
    embedding API to build it when it is missing or incomplete."""
    products = products if products is not None else PRODUCTS
    existing = get_chroma_client(persist_path).get_or_create_collection(name=collection_name)
    if existing.count() >= len(products):
        return existing
    return build_collection(embedding_client, persist_path, collection_name, products)


DEMO_QUERIES: list[tuple[str, str | None]] = [
    ("中高風險、適合積極投資、投資期間3年以上的基金產品", "Fund"),
    ("短期資金停泊，希望有流動性的定存", "TD"),
    ("美元資產配置，可承受一些匯率波動", "FX"),
]


def main() -> None:
    """Manual demo: index all 30 products, then run a few sample searches.

    Usage:
        python src/rag/retrieval.py                          # 3 built-in demo queries (one per category)
        python src/rag/retrieval.py "查詢文字"                  # custom query, no category filter
        python src/rag/retrieval.py "查詢文字" Fund              # custom query, filtered to one category
    """
    load_env_file()

    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY not set.\n"
            f"Add it to {Path(__file__).resolve().parents[2] / '.env'} (see .env.example), "
            "or get a key at https://aistudio.google.com/apikey"
        )
        raise SystemExit(1)

    client = GeminiEmbeddingClient()

    print(f"Indexing {len(PRODUCTS)} products -> {CHROMA_PATH}")
    collection = build_collection(client)
    print(f"Collection count: {collection.count()}\n")

    if len(sys.argv) > 1:
        query = sys.argv[1]
        category = sys.argv[2] if len(sys.argv) > 2 else None
        queries = [(query, category)]
    else:
        queries = DEMO_QUERIES

    for query, category in queries:
        label = f" (category={category})" if category else ""
        print(f"Query: {query}{label}")
        hits = search(collection, client, query,
                      product_category=category, n_results=3)
        for h in hits:
            print(
                f"  {h['product_id']:10s} | risk={h['metadata']['risk_level']:6s} | "
                f"distance={h['distance']:.4f} | {h['document'][:50]}"
            )
        print()


if __name__ == "__main__":
    main()
