from __future__ import annotations

"""
表结构定义：从 `config/data_schema.yaml` 中加载 SQL 表定义，
并在 Python 中提供：
- 列名集合；
- 主键列名；
- 可 safely 用于 IR→SQL 的字段判断。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TableSchema:
    name: str
    columns: List[str]
    primary_key: Optional[str]

    def has_column(self, col: str) -> bool:
        return col in self.columns


class DBSchema:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        schema_path = project_root / "config" / "data_schema.yaml"
        with schema_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._tables: Dict[str, TableSchema] = {}
        for name, t_cfg in cfg.get("sql_tables", {}).items():
            cols_def = t_cfg.get("columns", {})
            columns = list(cols_def.keys())
            pk = None
            # 简单解析 PRIMARY KEY 关键字
            for col_name, decl in cols_def.items():
                if isinstance(decl, str) and "PRIMARY KEY" in decl.upper():
                    pk = col_name
                    break

            self._tables[name] = TableSchema(
                name=name, columns=columns, primary_key=pk
            )

        logger.info("db_schema_loaded", tables=list(self._tables.keys()))

    def get(self, table: str) -> Optional[TableSchema]:
        return self._tables.get(table)

    def ensure_table(self, table: str) -> TableSchema:
        ts = self.get(table)
        if ts is None:
            raise ValueError(f"unknown table: {table}")
        return ts


__all__ = ["DBSchema", "TableSchema"]

