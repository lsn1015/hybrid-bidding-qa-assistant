from __future__ import annotations

"""
向量库封装（FAISS）。

职责：
- 为 policy/opinion 等集合提供 add / search 接口；
- 维护 id -> 原文 / 元数据 的映射（内存字典或简单持久化）。

注意：这里实现的是本地 FAISS in-memory 版本，真正的持久化、
多进程共享可按实际部署环境扩展。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np
import yaml

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: Dict


class VectorStore:
    def __init__(self, collection_name: str, dim: int) -> None:
        self.collection_name = collection_name
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._records: Dict[int, VectorRecord] = {}
        self._next_idx = 0

    # ------------------------------------------------------------------ #
    # Data operations
    # ------------------------------------------------------------------ #
    def add(self, embeddings: Iterable[List[float]], records: Iterable[VectorRecord]) -> None:
        embs = np.asarray(list(embeddings), dtype="float32")
        if embs.size == 0:
            return
        if embs.shape[1] != self.dim:
            raise ValueError(f"embedding dim mismatch: expected {self.dim}, got {embs.shape[1]}")
        start_id = self._next_idx
        self._index.add(embs)
        for offset, rec in enumerate(records):
            self._records[start_id + offset] = rec
        self._next_idx += embs.shape[0]
        logger.info(
            "vector_store_add",
            collection=self.collection_name,
            added=embs.shape[0],
            total=self._next_idx,
        )

    def search(self, query_embeddings: Iterable[List[float]], top_k: int) -> List[List[Tuple[VectorRecord, float]]]:
        q = np.asarray(list(query_embeddings), dtype="float32")
        if q.size == 0 or self._index.ntotal == 0:
            return [[] for _ in range(len(q))]
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim mismatch: expected {self.dim}, got {q.shape[1]}")

        sims, ids = self._index.search(q, top_k)
        results: List[List[Tuple[VectorRecord, float]]] = []
        for row_ids, row_sims in zip(ids, sims):
            row: List[Tuple[VectorRecord, float]] = []
            for idx, score in zip(row_ids, row_sims):
                if idx == -1:
                    continue
                rec = self._records.get(int(idx))
                if rec is not None:
                    row.append((rec, float(score)))
            results.append(row)
        return results


def build_vector_store_from_config(collection_key: str) -> VectorStore:
    """
    根据 `config/data_schema.yaml` 中 vector_collections 的定义构建 VectorStore。
    """
    project_root = Path(__file__).resolve().parents[1]
    schema_path = project_root / "config" / "data_schema.yaml"
    with schema_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    coll_cfg = cfg.get("vector_collections", {}).get(collection_key, {})
    dim = int(coll_cfg.get("embedding_dim", 1024))
    logger.info("init_vector_store", collection=collection_key, dim=dim)
    return VectorStore(collection_name=collection_key, dim=dim)


__all__ = ["VectorStore", "VectorRecord", "build_vector_store_from_config"]

