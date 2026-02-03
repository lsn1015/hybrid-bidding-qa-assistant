from __future__ import annotations

"""
来源引用信息构建。

将 RAG 文本文档与 SQL 行转换为前端可展示的 citation 列表，包含：
- type: rag/sql
- id / title / url(optional)
- score 或其它辅助信息
"""

from typing import Any, Dict, List, Optional

from ..retrieval.reranker import RerankResult


def build_rag_citations(docs: List[RerankResult]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for d in docs:
        meta = d.metadata or {}
        title = meta.get("policy_title") or meta.get("title") or meta.get("doc_id") or d.id
        citations.append(
            {
                "type": "rag",
                "id": d.id,
                "title": title,
                "score": d.score,
                "source_web": meta.get("source_web"),
                "publish_date": meta.get("publish_date"),
            }
        )
    return citations


def build_sql_citations(
    table: str, rows: List[Dict[str, Any]], pk_field: Optional[str] = None
) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for row in rows:
        pk = row.get(pk_field) if pk_field else None
        citations.append(
            {
                "type": "sql",
                "table": table,
                "primary_key": pk,
            }
        )
    return citations


__all__ = ["build_rag_citations", "build_sql_citations"]

