from __future__ import annotations

"""
标准化时间 / 金额 / 电话等字段。

规则来自 `config/rules.yaml` 中 `normalize` 段：
- date_format: "%Y-%m-%d"
- amount.unit: CNY
- amount.max_value / min_value
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import yaml

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NormalizeConfig:
    date_format: str
    amount_unit: str
    amount_max: float
    amount_min: float


def _load_config() -> NormalizeConfig:
    project_root = Path(__file__).resolve().parents[1]
    rules_path = project_root / "config" / "rules.yaml"

    with rules_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n = cfg.get("normalize", {})
    amount_cfg = n.get("amount", {})
    return NormalizeConfig(
        date_format=n.get("date_format", "%Y-%m-%d"),
        amount_unit=amount_cfg.get("unit", "CNY"),
        amount_max=float(amount_cfg.get("max_value", 1e9)),
        amount_min=float(amount_cfg.get("min_value", 0.0)),
    )


_CFG = _load_config()


def normalize_date(text: str) -> Optional[str]:
    """
    将日期字符串标准化为 YYYY-MM-DD。

    支持：
    - 2024年3月15日
    - 2024-03-15
    - 2024/03/15
    - 2024年3月  -> 默认补 1 日
    """
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime(_CFG.date_format)
        except ValueError:
            continue

    # 处理“2024年3月15日”、“2024年3月”
    import re

    m = re.match(
        r"(?P<year>\d{4})[年\-/\.](?P<month>\d{1,2})(?:[月\-/\.](?P<day>\d{1,2})[日号]?)?",
        text,
    )
    if not m:
        logger.debug("normalize_date failed", text=text)
        return None

    year = int(m.group("year"))
    month = int(m.group("month"))
    day = int(m.group("day") or 1)
    try:
        dt = date(year, month, day)
    except ValueError:
        logger.debug("normalize_date invalid date", text=text)
        return None
    return dt.strftime(_CFG.date_format)


def normalize_amount(text: str) -> Optional[float]:
    """
    将金额字符串归一为“元”为单位的浮点数。

    支持：
    - 500万元
    - 3000 万
    - 1.2亿元
    - 8000元
    """
    import re

    text = text.replace(",", "").strip()
    m = re.match(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>亿元|亿|万元|万|元)", text
    )
    if not m:
        logger.debug("normalize_amount failed", text=text)
        return None

    value = float(m.group("value"))
    unit = m.group("unit")

    if unit in ("亿元", "亿"):
        value *= 1e8
    elif unit in ("万元", "万"):
        value *= 1e4
    # 元 不需转换

    if value < _CFG.amount_min or value > _CFG.amount_max:
        logger.debug(
            "normalize_amount out_of_range",
            value=value,
            min=_CFG.amount_min,
            max=_CFG.amount_max,
        )
        return None
    return value


def normalize_phone(text: str) -> Optional[str]:
    """
    对手机号做简单标准化：
    - 去除空格和 `+86` 前缀；
    """
    import re

    text = text.strip()
    m = re.search(r"(1[3-9]\d{9})", text)
    if not m:
        return None
    return m.group(1)


__all__ = ["NormalizeConfig", "normalize_date", "normalize_amount", "normalize_phone"]

