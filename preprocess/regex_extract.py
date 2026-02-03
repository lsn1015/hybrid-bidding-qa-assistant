from __future__ import annotations

"""
正则抽取（时间 / 金额 / 联系方式等字段）。

本模块侧重于**可结构化字段**的抽取，为：
- SQL 表（如 `tender_project.amount`, `tender_project.publish_date`）提供原始候选值；
- IR 结构中的 `structured_conditions` 提供线索；
- 归一化模块 `normalize` 提供原始字符串输入。
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from ..logging.logger import get_logger

logger = get_logger(__name__)


AMOUNT_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>亿元|亿|万元|万|元)",
    re.UNICODE,
)

DATE_PATTERN = re.compile(
    r"(?P<year>\d{4})[年\-/\.](?P<month>\d{1,2})[月\-/\.](?P<day>\d{1,2})[日号]?",
    re.UNICODE,
)

DATE_YM_PATTERN = re.compile(
    r"(?P<year>\d{4})[年\-/\.](?P<month>\d{1,2})[月]?",
    re.UNICODE,
)

PHONE_PATTERN = re.compile(
    r"(?:\+?86[-\s]?)?(1[3-9]\d{9})",
    re.UNICODE,
)


@dataclass
class RegexExtractResult:
    amounts: List[str]
    dates: List[str]
    phones: List[str]


def extract_amounts(text: str) -> List[str]:
    """
    抽取金额字符串，如：
    - 500万元
    - 3000 万
    - 1.2亿元
    """
    matches = [m.group(0).strip() for m in AMOUNT_PATTERN.finditer(text)]
    logger.debug("extract_amounts", count=len(matches))
    return matches


def extract_dates(text: str) -> List[str]:
    """
    抽取日期字符串，支持：
    - 2024年3月15日
    - 2024-03-15
    - 2024/03/15
    - 2024年3月
    """
    results: List[str] = []
    for m in DATE_PATTERN.finditer(text):
        results.append(m.group(0).strip())
    # 若全文没有精确到日的日期，则尝试只到“年-月”
    if not results:
        for m in DATE_YM_PATTERN.finditer(text):
            results.append(m.group(0).strip())
    logger.debug("extract_dates", count=len(results))
    return results


def extract_phones(text: str) -> List[str]:
    """
    抽取手机号（大陆 11 位）。
    """
    matches = [m.group(1) for m in PHONE_PATTERN.finditer(text)]
    logger.debug("extract_phones", count=len(matches))
    return matches


def extract_all(text: str) -> RegexExtractResult:
    """
    综合抽取金额 / 日期 / 电话。
    """
    return RegexExtractResult(
        amounts=extract_amounts(text),
        dates=extract_dates(text),
        phones=extract_phones(text),
    )


__all__ = [
    "RegexExtractResult",
    "extract_amounts",
    "extract_dates",
    "extract_phones",
    "extract_all",
]

