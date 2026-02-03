from __future__ import annotations

"""
TopK 语义检索。

负责：
- 从 Embedder 获取 query 向量；
- 调用 VectorStore 进行 TopK 检索；
- 输出统一的检索结果结构，供 Reranker 与 RAG 使用。
"""

from dataclasses import dataclass
from typing import Dict, List

from ..logging.logger import get_logger
from .embedder import Embedder
from .vector_store import VectorRecord, VectorStore, build_vector_store_from_config

logger = get_logger(__name__)


@dataclass
class RetrieveResult:
    id: str
    text: str
    metadata: Dict
    score: float


class Retriever:
    def __init__(self, collection_key: str, top_k: int) -> None:
        self._embedder = Embedder()
        self._store = build_vector_store_from_config(collection_key)
        self.top_k = top_k
        self.collection_key = collection_key

    def add_documents(self, docs: List[VectorRecord], embeddings: List[List[float]]) -> None:
        """
        预加载文档向量。
        """
        self._store.add(embeddings=embeddings, records=docs)

    def retrieve(self, query: str) -> List[RetrieveResult]:
        """
        对单条 query 做 TopK 语义检索。
        """
        q_emb = self._embedder.embed([query])
        results = self._store.search(q_emb, self.top_k)[0]
        out = [
            RetrieveResult(
                id=rec.id,
                text=rec.text,
                metadata=rec.metadata,
                score=score,
            )
            for rec, score in results
        ]
        logger.info(
            "retrieve",
            collection=self.collection_key,
            top_k=self.top_k,
            returned=len(out),
        )
        return out


__all__ = ["Retriever", "RetrieveResult"]

