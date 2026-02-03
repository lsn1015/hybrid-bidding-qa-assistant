from __future__ import annotations

"""
FastAPI 输入输出 schema 定义。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户自然语言问题")
    user_role: Optional[str] = Field(
        default=None, description="用户角色，如 '采购', '销售', '投资人' 等"
    )
    debug: bool = Field(
        default=False, description="是否返回调试信息（IR、SQL、上下文等）"
    )


class Citation(BaseModel):
    type: str
    id: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None
    table: Optional[str] = None
    primary_key: Optional[Any] = None
    source_web: Optional[str] = None
    publish_date: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    route: str
    citations: List[Citation] = []
    debug: Optional[Dict[str, Any]] = None


__all__ = ["QueryRequest", "QueryResponse", "Citation"]

