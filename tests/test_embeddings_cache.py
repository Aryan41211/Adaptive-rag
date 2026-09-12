"""
Tests for the bounded embedding-query cache.

The cache exists because one turn embeds the same question text three or four
times (classifier, retriever, citations). These tests pin the caching
behaviour directly against a counting fake, without any network access.
"""

import threading

from src.llms.openai import CachedEmbeddings


class CountingEmbedding:
    """A deterministic embedding that counts every call."""

    def __init__(self, size: int = 8):
        self.size = size
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [float(len(text)) + i for i in range(self.size)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self.embed_query(text) for text in texts]


def test_repeated_query_is_embedded_once():
    fake = CountingEmbedding()
    cached = CachedEmbeddings(fake)

    first = cached.embed_query("what is the capital of France?")
    second = cached.embed_query("what is the capital of France?")
    third = cached.embed_query("what is the capital of France?")

    assert first == second == third
    assert fake.query_calls == 1


def test_distinct_queries_are_forwarded():
    fake = CountingEmbedding()
    cached = CachedEmbeddings(fake)

    cached.embed_query("question one")
    cached.embed_query("question two")

    assert fake.query_calls == 2


def test_evicted_entries_are_embedded_again():
    fake = CountingEmbedding()
    cached = CachedEmbeddings(fake, maxsize=2)

    cached.embed_query("a")
    cached.embed_query("b")
    cached.embed_query("c")
    assert fake.query_calls == 3

    cached.embed_query("b")
    assert fake.query_calls == 3

    cached.embed_query("a")
    assert fake.query_calls == 4


def test_document_embeddings_are_not_cached():
    fake = CountingEmbedding()
    cached = CachedEmbeddings(fake)

    cached.embed_documents(["alpha", "beta"])
    cached.embed_documents(["alpha", "beta"])

    assert fake.document_calls == 2


def test_cache_is_thread_safe():
    fake = CountingEmbedding()
    cached = CachedEmbeddings(fake)

    results: list[list[float]] = []
    errors: list[Exception] = []

    def worker():
        try:
            vector = cached.embed_query("same urgent question")
            for _ in range(50):
                assert cached.embed_query("same urgent question") == vector
            results.append(vector)
        except Exception as exc:  # noqa: BLE001 - surfaced on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert all(result == results[0] for result in results)
