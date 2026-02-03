from __future__ import annotations

"""
置信度 / 不确定性判断。

混合 RAG + SQL 场景中，需要对：
  - 向量检索得分；
  - reranker 分数；
  - SQL 命中行数；
做统一的置信度评估，并在低置信情况下触发
兜底话术（见 `config/rules.yaml` 中 answer.uncertainty_response）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfidenceInput:
    retriever_scores: Optional[List[float]] = None
    reranker_scores: Optional[List[float]] = None
    sql_row_count: Optional[int] = None


@dataclass
class ConfidenceResult:
    score: float
    is_confident: bool


class ConfidenceChecker:
    """
    将多源信号压缩为一个统一置信度分数 [0, 1]。

    当前策略（可按需替换为更复杂的统计/ML 模型）：
    - 向量检索：用 top1 相似度（若存在）；
    - reranker：用 top1 分数（若存在）；
    - SQL：根据命中行数做一个简单的 logistic 映射；
    - 最终置信度取三个信号的最大值；
    - 与 router.confidence_threshold 对比判定是否置信。
    """

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        rules_path = project_root / "config" / "rules.yaml"

        with rules_path.open("r", encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)

        self.threshold: float = float(
            self._rules.get("router", {}).get("confidence_threshold", 0.7)
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(self, inputs: ConfidenceInput) -> ConfidenceResult:
        scores: List[float] = []

        if inputs.retriever_scores:
            scores.append(max(inputs.retriever_scores))

        if inputs.reranker_scores:
            scores.append(max(inputs.reranker_scores))

        if inputs.sql_row_count is not None:
            # 简单的：命中 0 行 → 0.0，命中 >= 50 行 → ~1.0
            count = max(0, inputs.sql_row_count)
            sql_score = min(1.0, count / 50.0)
            scores.append(sql_score)

        final_score = max(scores) if scores else 0.0
        is_confident = final_score >= self.threshold

        logger.debug(
            "Confidence evaluation: scores=%s, final=%.3f, threshold=%.3f, is_confident=%s",
            scores,
            final_score,
            self.threshold,
            is_confident,
        )

        return ConfidenceResult(score=final_score, is_confident=is_confident)


__all__ = ["ConfidenceChecker", "ConfidenceInput", "ConfidenceResult"]

