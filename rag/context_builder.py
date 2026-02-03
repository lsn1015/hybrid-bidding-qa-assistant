from __future__ import annotations

"""
RAG 上下文拼装。

职责：
- 将检索到的 policy/opinion 文本与结构化 SQL 结果合并为一个上下文字符串；
- 控制总 token/字符长度（参考 `config/rules.yaml` 中 rag.max_context_tokens）；
- 为 AnswerGenerator 提供干净、带 source tag 的上下文。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..logging.logger import get_logger
from ..retrieval.reranker import RerankResult

logger = get_logger(__name__)


@dataclass
class BuiltContext:
    text: str
    sources: List[Dict[str, Any]]


class ContextBuilder:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        rules_path = project_root / "config" / "rules.yaml"
        with rules_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        rag_cfg = cfg.get("rag", {})
        # 简单用字符数近似 token 数
        self.max_tokens: int = int(rag_cfg.get("max_context_tokens", 1200))

    def build(
        self,
        rag_docs: Optional[List[RerankResult]] = None,
        sql_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> BuiltContext:
        rag_docs = rag_docs or []
        sql_rows = sql_rows or []

        parts: List[str] = []
        sources: List[Dict[str, Any]] = []

        # 1) 结构化 SQL 结果
        if sql_rows:
            parts.append("【结构化数据】")
            for i, row in enumerate(sql_rows):
                parts.append(f"- 记录 {i+1}: {row}")
            sources.append({"type": "sql", "count": len(sql_rows)})

        # 2) RAG 文本
        if rag_docs:
            parts.append("【非结构化文本】")
            for i, doc in enumerate(rag_docs):
                meta = doc.metadata or {}
                title = meta.get("policy_title") or meta.get("title") or meta.get("doc_id") or doc.id
                header = f"[{i+1}] 标题: {title} | 分数: {doc.score:.3f}"
                parts.append(header)
                parts.append(doc.text.strip())
                sources.append(
                    {
                        "type": "rag",
                        "id": doc.id,
                        "score": doc.score,
                        "metadata": meta,
                    }
                )

        full = "\n".join(parts)

        # 3) 截断到最大“token”数（这里用字符近似）
        if len(full) > self.max_tokens:
            full = full[: self.max_tokens] + "\n……（内容已截断）"

        logger.info(
            "build_context",
            rag_docs=len(rag_docs),
            sql_rows=len(sql_rows),
            length=len(full),
        )
        return BuiltContext(text=full, sources=sources)


__all__ = ["ContextBuilder", "BuiltContext"]

