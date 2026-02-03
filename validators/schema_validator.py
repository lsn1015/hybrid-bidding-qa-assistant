from __future__ import annotations

"""
IR schema 校验。

根据 `config/data_schema.yaml` 与 `config/rules.yaml` 中对
`intermediate_representations.unified_query_ir` 和 `ir` 的定义，
对 LLM 生成的 IR 做结构与字段合法性校验。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaValidationResult:
    ok: bool
    errors: List[str]

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("IR schema validation failed: " + "; ".join(self.errors))


class IRSchemaValidator:
    """
    校验 unified_query_ir：

    - 顶层必备字段是否存在；
    - 字段类型是否与 schema 定义兼容（宽松检查）；
    - 过滤条件是否满足 `rules.yaml` 中的约束。
    """

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data_schema_path = project_root / "config" / "data_schema.yaml"
        rules_path = project_root / "config" / "rules.yaml"

        with data_schema_path.open("r", encoding="utf-8") as f:
            self._data_schema = yaml.safe_load(f)

        with rules_path.open("r", encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)

        ir_cfg = self._rules.get("ir", {})
        self.required_fields: List[str] = ir_cfg.get("required_fields", [])
        self.max_filters: int = int(ir_cfg.get("max_filters", 5))
        self.allowed_ops: List[str] = list(ir_cfg.get("allowed_ops", []))
        self.allow_fuzzy: bool = bool(ir_cfg.get("allow_fuzzy", False))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def validate(self, ir: Dict[str, Any]) -> SchemaValidationResult:
        errors: List[str] = []

        # 1) 顶层必需字段
        for field in self.required_fields:
            if field not in ir:
                errors.append(f"missing required field: {field}")

        # 2) query_type 合法性
        query_type = ir.get("query_type")
        allowed_query_types = (
            self._data_schema.get("intermediate_representations", {})
            .get("unified_query_ir", {})
            .get("schema", {})
            .get("query_type", "")
        )
        # e.g. "ENUM[rag, sql, hybrid]"
        enum_values = self._parse_enum(allowed_query_types)
        if enum_values and query_type not in enum_values:
            errors.append(f"invalid query_type: {query_type}, allowed={enum_values}")

        # 3) filters 校验
        filters = ir.get("filters", [])
        if not isinstance(filters, list):
            errors.append("filters must be a list")
        else:
            if len(filters) > self.max_filters:
                errors.append(
                    f"too many filters: {len(filters)} > max_filters={self.max_filters}"
                )
            for idx, cond in enumerate(filters):
                self._validate_single_filter(cond, idx, errors)

        ok = len(errors) == 0
        if ok:
            logger.debug("IR schema validation passed.")
        else:
            logger.warning("IR schema validation failed: %s", "; ".join(errors))
        return SchemaValidationResult(ok=ok, errors=errors)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_enum(enum_decl: str) -> List[str]:
        """
        将 "ENUM[a, b, c]" 解析为 ["a", "b", "c"]。
        """
        if not isinstance(enum_decl, str) or not enum_decl.startswith("ENUM["):
            return []
        inner = enum_decl[len("ENUM[") : -1]
        return [v.strip() for v in inner.split(",") if v.strip()]

    def _validate_single_filter(
        self, cond: Dict[str, Any], idx: int, errors: List[str]
    ) -> None:
        field = cond.get("field")
        op = cond.get("op")

        if not field:
            errors.append(f"filters[{idx}].field is required")
        if not op:
            errors.append(f"filters[{idx}].op is required")
        elif op not in self.allowed_ops:
            errors.append(
                f"filters[{idx}].op '{op}' is not in allowed_ops={self.allowed_ops}"
            )

        # between 操作需要 value 为长度2的 list
        if op == "between":
            value = cond.get("value")
            if not (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(v is not None for v in value)
            ):
                errors.append(
                    f"filters[{idx}].value for 'between' must be 2-element list/tuple"
                )

        if op == "like" and not self.allow_fuzzy:
            errors.append("fuzzy search 'like' is disabled by rules.yaml")


def validate_ir_schema(ir: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    便捷函数，供其他模块调用。
    """
    validator = IRSchemaValidator()
    result = validator.validate(ir)
    return result.ok, result.errors


__all__ = ["IRSchemaValidator", "SchemaValidationResult", "validate_ir_schema"]

