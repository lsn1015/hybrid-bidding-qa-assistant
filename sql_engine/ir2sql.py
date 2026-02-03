from __future__ import annotations

"""
IR → SQL 映射。

输入：
- IR（字典），需包含 table / select / filters 等字段；

输出：
- 安全的 SELECT 语句 + 参数列表，遵守：
  - 只允许 `rules.yaml` 中 `allowed_ops`；
  - 不允许访问 schema 未定义的字段；
  - 自动插入安全的 LIMIT（不超过 `rules.sql.max_limit`）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ..logging.logger import get_logger
from .db_schema import DBSchema

logger = get_logger(__name__)


@dataclass
class SqlBuildResult:
    sql: str
    params: List[Any]


class IR2SQL:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        rules_path = project_root / "config" / "rules.yaml"
        with rules_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._ir_rules = cfg.get("ir", {})
        self._sql_rules = cfg.get("sql", {})
        self._schema = DBSchema()

        self.allowed_ops: List[str] = list(self._ir_rules.get("allowed_ops", []))
        self.max_limit: int = int(self._sql_rules.get("max_limit", 100))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, ir: Dict[str, Any]) -> SqlBuildResult:
        table_name = ir.get("table")
        if not table_name:
            raise ValueError("IR missing 'table' field for SQL generation.")
        table_schema = self._schema.ensure_table(table_name)

        select_cols = ir.get("select") or ["*"]
        if select_cols == ["*"]:
            cols_sql = "*"
        else:
            # 只允许 schema 中存在的列
            invalid = [c for c in select_cols if not table_schema.has_column(c)]
            if invalid:
                raise ValueError(f"invalid select columns: {invalid}")
            cols_sql = ", ".join(f"`{c}`" for c in select_cols)

        where_sql_parts: List[str] = []
        params: List[Any] = []

        for cond in ir.get("filters") or []:
            field = cond.get("field")
            op = cond.get("op")
            value = cond.get("value")

            if not field or not op:
                continue
            if op not in self.allowed_ops:
                logger.warning("skip disallowed op", op=op)
                continue
            if not table_schema.has_column(field):
                logger.warning("skip unknown column", field=field, table=table_name)
                continue

            if op == "between":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                where_sql_parts.append(f"`{field}` BETWEEN %s AND %s")
                params.extend([value[0], value[1]])
            elif op == "like":
                where_sql_parts.append(f"`{field}` LIKE %s")
                params.append(str(value))
            else:
                # 其它二元 op
                where_sql_parts.append(f"`{field}` {op} %s")
                params.append(value)

        where_sql = ""
        if where_sql_parts:
            where_sql = " WHERE " + " AND ".join(where_sql_parts)

        limit = ir.get("limit")
        if limit is None or not isinstance(limit, int) or limit <= 0:
            limit = self.max_limit
        else:
            limit = min(limit, self.max_limit)

        sql = f"SELECT {cols_sql} FROM {table_name}{where_sql} LIMIT {limit}"
        logger.info("ir_to_sql", table=table_name, limit=limit)
        return SqlBuildResult(sql=sql, params=params)


__all__ = ["IR2SQL", "SqlBuildResult"]

