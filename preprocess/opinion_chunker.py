from __future__ import annotations

"""
舆情信息没有严格的结构化标准，这里按**等长+句子边界优先**的方式切分。

输出结构对应 `config/data_schema.yaml` 中 `opinion_chunks` 集合。
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from ..logging.logger import get_logger
from .normalize import normalize_date

logger = get_logger(__name__)


SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?]\s*")


@dataclass
class OpinionChunk:
    id: str
    doc_id: str
    chunk_index: int
    total_chunks: int
    title: str
    source_web: str
    publish_date: Optional[str]
    related_company: Optional[str]
    sentiment: str
    risk_type: str
    text: str


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT_PATTERN.split(text) if p.strip()]
    return parts or [text.strip()]


def chunk_opinion_text(
    doc_id: str,
    title: str,
    source_web: str,
    raw_text: str,
    publish_date: Optional[str] = None,
    related_company: Optional[str] = None,
    sentiment: str = "unknown",
    risk_type: str = "unknown",
    max_chars: int = 500,
) -> List[OpinionChunk]:
    """
    将舆情文本切分为长度约为 max_chars 的块，尽量在句子边界切断。
    """
    sentences = _split_sentences(raw_text)
    acc = ""
    pieces: List[str] = []
    for s in sentences:
        if len(acc) + len(s) <= max_chars:
            acc += (s + "。")
        else:
            if acc.strip():
                pieces.append(acc.strip())
            acc = s + "。"
    if acc.strip():
        pieces.append(acc.strip())

    total = len(pieces)
    norm_date = normalize_date(publish_date) if publish_date else None

    chunks: List[OpinionChunk] = []
    for idx, piece in enumerate(pieces):
        chunk_id = f"opinion_{doc_id}_{idx:02d}"
        chunks.append(
            OpinionChunk(
                id=chunk_id,
                doc_id=doc_id,
                chunk_index=idx,
                total_chunks=total,
                title=title,
                source_web=source_web,
                publish_date=norm_date,
                related_company=related_company,
                sentiment=sentiment,
                risk_type=risk_type,
                text=piece,
            )
        )

    logger.info(
        "chunk_opinion_text",
        doc_id=doc_id,
        title=title,
        total_chunks=total,
    )
    return chunks


def chunk_opinion_file(
    input_path: Path,
    output_path: Path,
    doc_id: Optional[str] = None,
    title: Optional[str] = None,
    source_web: str = "unknown",
    publish_date: Optional[str] = None,
    related_company: Optional[str] = None,
    sentiment: str = "unknown",
    risk_type: str = "unknown",
    max_chars: int = 500,
) -> None:
    text = input_path.read_text(encoding="utf-8")
    doc_id = doc_id or input_path.stem
    title = title or input_path.stem
    chunks = chunk_opinion_text(
        doc_id=doc_id,
        title=title,
        source_web=source_web,
        raw_text=text,
        publish_date=publish_date,
        related_company=related_company,
        sentiment=sentiment,
        risk_type=risk_type,
        max_chars=max_chars,
    )
    data = [asdict(c) for c in chunks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    logger.info(
        "chunk_opinion_file",
        input=str(input_path),
        output=str(output_path),
        chunks=len(chunks),
    )


__all__ = ["OpinionChunk", "chunk_opinion_text", "chunk_opinion_file"]

