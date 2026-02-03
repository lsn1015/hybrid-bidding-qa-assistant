from __future__ import annotations

"""
TOP K -> TOP N 二次检索（Reranker）。

使用 cross-encoder（如 `BAAI/bge-reranker-v2-m3`），
对 (query, document) 做打分，并按分数截断为 TopN。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

try:  # 可选依赖
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

from ..logging.logger import get_logger
from .retriever import RetrieveResult

logger = get_logger(__name__)


@dataclass
class RerankResult:
    id: str
    text: str
    metadata: dict
    score: float


class Reranker:
    def __init__(self, top_n: int) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model_cfg_path = project_root / "config" / "model.yaml"

        with model_cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        rerank_cfg = cfg.get("rerank", {})
        self.model_name: str = rerank_cfg.get(
            "model_name", "BAAI/bge-reranker-v2-m3"
        )
        self.top_n = top_n

        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers CrossEncoder not available, please `pip install sentence-transformers`."
            )

        logger.info("loading_reranker_model", model=self.model_name)
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[RetrieveResult]) -> List[RerankResult]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        scored = [
            (candidates[i], float(scores[i])) for i in range(len(candidates))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        top = scored[: self.top_n]
        out = [
            RerankResult(
                id=c.id,
                text=c.text,
                metadata=c.metadata,
                score=score,
            )
            for c, score in top
        ]
        logger.info(
            "rerank",
            total=len(candidates),
            top_n=self.top_n,
            returned=len(out),
        )
        return out


__all__ = ["Reranker", "RerankResult"]

