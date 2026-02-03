from __future__ import annotations

"""
离线预处理入口脚本。

功能：
1. 扫描 data/raw/policy、data/raw/opinion 下的原始文件（txt / pdf）；
2. 调用 policy_chunker / opinion_chunker 生成分块；
3. 汇总写入：
   - data/processed/policy_chunks.json
   - data/processed/opinion_chunks.json

注意：
- PDF 文本抽取依赖 pdfplumber，可通过 `pip install pdfplumber` 安装；
- 该脚本仅负责文本分块与结构化中间结果的生成，不直接写入数据库或构建向量索引。
"""

import json
from pathlib import Path
from typing import List

try:  # 可选依赖
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

from preprocess.policy_chunker import PolicyChunk, chunk_policy_text
from preprocess.opinion_chunker import OpinionChunk, chunk_opinion_text
from logging.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        if pdfplumber is None:
            logger.warning(
                "skip_pdf_no_pdfplumber", file=str(path)
            )
            return ""
        text = []
        with pdfplumber.open(path) as pdf:  # type: ignore
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    logger.warning("skip_unsupported_file", file=str(path))
    return ""


def preprocess_policy() -> None:
    policy_dir = RAW_DIR / "policy"
    if not policy_dir.exists():
        logger.warning("policy_dir_not_found", dir=str(policy_dir))
        return

    all_chunks: List[PolicyChunk] = []
    for path in sorted(policy_dir.glob("*")):
        if not path.is_file():
            continue
        text = _read_text(path)
        if not text.strip():
            continue
        doc_id = path.stem
        title = path.stem
        chunks = chunk_policy_text(
            doc_id=doc_id,
            title=title,
            source_web="unknown",
            raw_text=text,
            publish_date=None,
        )
        all_chunks.extend(chunks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "policy_chunks.json"
    data = [c.__dict__ for c in all_chunks]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    logger.info("preprocess_policy_done", count=len(all_chunks), output=str(out_path))


def preprocess_opinion() -> None:
    opinion_dir = RAW_DIR / "opinion"
    if not opinion_dir.exists():
        logger.warning("opinion_dir_not_found", dir=str(opinion_dir))
        return

    all_chunks: List[OpinionChunk] = []
    for path in sorted(opinion_dir.glob("*")):
        if not path.is_file():
            continue
        text = _read_text(path)
        if not text.strip():
            continue
        doc_id = path.stem
        title = path.stem
        chunks = chunk_opinion_text(
            doc_id=doc_id,
            title=title,
            source_web="unknown",
            raw_text=text,
            publish_date=None,
            related_company=None,
        )
        all_chunks.extend(chunks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "opinion_chunks.json"
    data = [c.__dict__ for c in all_chunks]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    logger.info("preprocess_opinion_done", count=len(all_chunks), output=str(out_path))


def main() -> None:
    logger.info("preprocess_main_start")
    preprocess_policy()
    preprocess_opinion()
    logger.info("preprocess_main_finished")


if __name__ == "__main__":
    main()

