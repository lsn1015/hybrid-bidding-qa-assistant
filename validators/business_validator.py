from __future__ import annotations

"""
业务规则校验。

结合 `config/rules.yaml` 与 `config/data_schema.yaml` 中的约束，
对金额、时间范围、分页大小等做安全与业务合理性判断。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BusinessValidationResult:
    ok: bool
    warnings: List[str]


class BusinessValidator:
    """
    针对统一 IR 或 SQL 查询参数进行业务侧校验。

    当前实现：
    - 限制金额区间不能超过配置中的 max_value；
    - 限制 amount >= 0；
    - 限制日期范围不能为 publish_start > publish_end；
    - 预留 hook，可扩展更多行业规则（如：公司注册资本合理区间等）。
    """

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        rules_path = project_root / "config" / "rules.yaml"

        with rules_path.open("r", encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)

        normalize_cfg = self._rules.get("normalize", {})
        amount_cfg = normalize_cfg.get("amount", {})

        self.amount_max: float = float(amount_cfg.get("max_value", 1e9))
        self.amount_min: float = float(amount_cfg.get("min_value", 0.0))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def validate_ir(self, ir: Dict[str, Any]) -> BusinessValidationResult:
        warnings: List[str] = []

        # 1) 金额区间（若存在）
        amount_range: Optional[Dict[str, Any]] = (
            ir.get("structured_conditions", {}).get("amount_range")  # type: ignore[assignment]
        )
        if isinstance(amount_range, dict):
            min_v = amount_range.get("min")
            max_v = amount_range.get("max")
            if min_v is not None and min_v < self.amount_min:
                warnings.append(
                    f"amount.min={min_v} < configured min_value={self.amount_min}, clipped."
                )
            if max_v is not None and max_v > self.amount_max:
                warnings.append(
                    f"amount.max={max_v} > configured max_value={self.amount_max}, clipped."
                )
            if min_v is not None and max_v is not None and min_v > max_v:
                warnings.append("amount_range.min > amount_range.max, will be swapped.")

        # 2) 日期范围（若存在），这里只做简单的逻辑检查，不解析日期格式
        date_range: Optional[Dict[str, Any]] = (
            ir.get("structured_conditions", {}).get("date_range")  # type: ignore[assignment]
        )
        if isinstance(date_range, dict):
            start = date_range.get("publish_start")
            end = date_range.get("publish_end")
            if start and end and str(start) > str(end):
                warnings.append("date_range.publish_start > publish_end.")

        ok = len(warnings) == 0
        if ok:
            logger.debug("Business validation passed.")
        else:
            logger.info("Business validation warnings: %s", "; ".join(warnings))

        return BusinessValidationResult(ok=ok, warnings=warnings)


__all__ = ["BusinessValidator", "BusinessValidationResult"]

