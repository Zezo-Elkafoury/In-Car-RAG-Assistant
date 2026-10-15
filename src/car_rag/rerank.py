"""Reranking via the OpenRouter /rerank endpoint (e.g. cohere/rerank-v3.5)."""

import requests
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import Field


class OpenRouterReranker(BaseDocumentCompressor):
    """Cross-encoder reranker backed by OpenRouter's /rerank API."""

    api_key: str = Field(repr=False)
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "cohere/rerank-v3.5"
    top_n: int = 3
    timeout: int = 60

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks=None,
    ) -> list[Document]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(self.top_n, len(documents)),
        }
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        except Exception as exc:  # noqa: BLE001 - surface API errors with context
            raise RuntimeError(f"OpenRouter rerank request failed: {exc}") from exc

        reranked: list[Document] = []
        for item in sorted(results, key=lambda r: r["relevance_score"], reverse=True):
            doc = documents[item["index"]]
            doc.metadata = {**doc.metadata, "rerank_score": item["relevance_score"]}
            reranked.append(doc)
        return reranked
