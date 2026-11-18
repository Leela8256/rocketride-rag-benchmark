"""Vector store write for document-pdf-rag-v1. No fastapi, no langgraph imports.

LangChain's own Postgres vector store, used the way its documentation shows:

    PGEngine.from_connection_string(url)              one engine for the process
    engine.init_vectorstore_table(table, 384)         the library's default table
    PGVectorStore.create_sync(engine, table, embedding_service)
    store.add_embeddings(texts, vectors, metadatas)   the library's write for vectors already computed
    store.similarity_search_with_score_by_vector(vector, k)   the library's search for a vector already computed

Nothing is tuned. The connection pool, the SQL, one commit per chunk and the absence of a vector index are all the
library's defaults (langchain-postgres 0.0.19). add_documents() would call the embedding service and then this same
add_embeddings(); the graph embeds in its own node, as it always has, so the vectors are passed in.

Connection settings come from the environment. The password is read here and is never logged or returned.
"""

import os
import threading
from typing import Any, Dict, List
from urllib.parse import quote

from workload.document.embed import EMBED_DIM, embed_chunks

TABLE = os.environ.get("RAG_PG_TABLE", "lc_chunks")
_store = None
_engine = None
_lock = threading.Lock()


def _url() -> str:
    env = os.environ
    return "postgresql+psycopg://%s:%s@%s:%s/%s" % (
        quote(env["RAG_PG_USER"], safe=""),
        quote(env["RAG_PG_PASSWORD"], safe=""),
        env["RAG_PG_HOST"],
        env["RAG_PG_PORT"],
        quote(env["RAG_PG_DATABASE"], safe=""),
    )


def _embedding_service():
    from langchain_core.embeddings import Embeddings

    class SharedModelEmbeddings(Embeddings):
        """The embedding service the store is created with: the same shared model the embed node calls."""

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return embed_chunks(list(texts))

        def embed_query(self, text: str) -> List[float]:
            return embed_chunks([text])[0]

    return SharedModelEmbeddings()


def load_store():
    """Create (once) the engine, the table and the store. Explicit, never lazy-in-node.

    The database starts empty for every run, so the table must not exist yet: a second create fails loudly.
    """
    global _store, _engine
    if _store is None:
        with _lock:
            if _store is None:
                from langchain_postgres import PGEngine, PGVectorStore

                engine = PGEngine.from_connection_string(url=_url())
                engine.init_vectorstore_table(table_name=TABLE, vector_size=EMBED_DIM)
                _engine = engine
                _store = PGVectorStore.create_sync(
                    engine=engine, table_name=TABLE, embedding_service=_embedding_service()
                )
    return _store


def is_loaded() -> bool:
    return _store is not None


def store_info() -> Dict[str, Any]:
    """What the store is, for the runtime report. Never includes the password or the connection string."""
    info: Dict[str, Any] = {
        "class": "langchain_postgres.PGVectorStore",
        "write": "add_embeddings",
        "table": TABLE,
        "vector_size": EMBED_DIM,
        "driver": "psycopg",
        "host": os.environ.get("RAG_PG_HOST"),
        "port": os.environ.get("RAG_PG_PORT"),
        "database": os.environ.get("RAG_PG_DATABASE"),
        "loaded": is_loaded(),
    }
    try:
        pool = _engine._pool.pool  # the SQLAlchemy pool behind the library's engine, left at its defaults
        info["pool"] = {"class": type(pool).__name__, "status": pool.status()}
    except Exception as exc:  # reporting only
        info["pool"] = "unavailable: %s" % type(exc).__name__
    return info


def search_chunks(vector: List[float], k: int) -> List[Dict[str, Any]]:
    """The k stored chunks nearest to a query vector, with LangChain's own retrieval call.

    similarity_search_with_score_by_vector() is the library's search for a vector that is already computed. It
    returns the documents with their cosine distance, nearest first. Nothing is filtered or re-ranked here.
    """
    found = load_store().similarity_search_with_score_by_vector(vector, k=k)
    return [{"text": doc.page_content, "source": (doc.metadata or {}).get("source"),
             "chunk": (doc.metadata or {}).get("chunk"), "distance": float(distance)} for doc, distance in found]


def store_chunks(source: str, chunks: List[Dict[str, Any]], vectors: List[List[float]]) -> int:
    """Write one document's chunks and vectors. Returns the number of rows the library reports written."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors differ in length: %d and %d" % (len(chunks), len(vectors)))
    ids = load_store().add_embeddings(
        texts=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": source, "chunk": c["index"]} for c in chunks],
    )
    return len(ids)
