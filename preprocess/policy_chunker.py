from __future__ import annotations

"""
政策类文件按照条/款/段结构化规则进行切分。

目标：
- 将长政策文件拆分为语义单元（条/款/段）；
- 生成结构化 JSON，便于后续向量化与存入 `policy_chunks` 集合。
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

from ..logging.logger import get_logger
from .normalize import normalize_date

logger = get_logger(__name__)


HEADING_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百千\d]+[条章节款])", re.MULTILINE
)


@dataclass
class PolicyChunk:
    id: str
    doc_id: str
    chunk_index: int
    total_chunks: int
    policy_title: str
    source_web: str
    publish_date: Optional[str]
    text: str


def _split_by_heading(text: str) -> List[str]:
    """
    按“第X条/章节/款”切分，保留标题到对应段落。
    """
    positions = [m.start() for m in HEADING_PATTERN.finditer(text)]
    if not positions:
        return [text.strip()] if text.strip() else []

    chunks = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        part = text[start:end].strip()
        if part:
            chunks.append(part)
    return chunks


def chunk_policy_text(
    doc_id: str,
    title: str,
    source_web: str,
    raw_text: str,
    publish_date: Optional[str] = None,
) -> List[PolicyChunk]:
    """
    对单个政策文本做切分，返回 PolicyChunk 列表。
    """
    pieces = _split_by_heading(raw_text)
    total = len(pieces)
    norm_date = normalize_date(publish_date) if publish_date else None

    chunks = []
    for idx, piece in enumerate(pieces):
        chunk_id = f"policy_{doc_id}_{idx:04d}"
        chunks.append(
            PolicyChunk(
                id=chunk_id,
                doc_id=doc_id,
                chunk_index=idx,
                total_chunks=total,
                policy_title=title,
                source_web=source_web,
                publish_date=norm_date,
                text=piece,
            )
        )
    logger.info(
        "chunk_policy_text",
        doc_id=doc_id,
        title=title,
        total_chunks=total,
    )
    return chunks


def chunk_policy_file(
    input_path: Path,
    output_path: Path,
    doc_id: Optional[str] = None,
    title: Optional[str] = None,
    source_web: str = "unknown",
    publish_date: Optional[str] = None,
) -> None:
    """
    从纯文本文件读取政策内容并写出 JSON 列表。
    """
    text = input_path.read_text(encoding="utf-8")
    doc_id = doc_id or input_path.stem
    title = title or input_path.stem
    chunks = chunk_policy_text(
        doc_id=doc_id,
        title=title,
        source_web=source_web,
        raw_text=text,
        publish_date=publish_date,
    )
    data = [asdict(c) for c in chunks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    logger.info(
        "chunk_policy_file",
        input=str(input_path),
        output=str(output_path),
        chunks=len(chunks),
    )


__all__ = ["PolicyChunk", "chunk_policy_text", "chunk_policy_file"]

