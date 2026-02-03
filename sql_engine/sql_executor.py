from __future__ import annotations

"""
只读 SQL 执行器。

根据：
- `config/database.yaml`：数据库连接信息；
- `config/rules.yaml`：允许的 SQL 命令（select）、禁用关键词、最大 limit；

执行安全的只读查询。
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import sqlalchemy
import yaml
from sqlalchemy.engine import Engine, Result

from ..logging.logger import get_logger

logger = get_logger(__name__)


class SqlExecutor:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        db_cfg_path = project_root / "config" / "database.yaml"
        rules_path = project_root / "config" / "rules.yaml"

        with db_cfg_path.open("r", encoding="utf-8") as f:
            db_cfg_all = yaml.safe_load(f)
        with rules_path.open("r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)

        db_cfg = db_cfg_all.get("database", {})
        self.type = db_cfg.get("type", "mysql")
        self.host = os.getenv("DB_HOST", db_cfg.get("host"))
        self.port = os.getenv("DB_PORT", db_cfg.get("port"))
        self.name = os.getenv("DB_NAME", db_cfg.get("name"))
        self.user = os.getenv("DB_USER", db_cfg.get("user"))
        self.password = os.getenv("DB_PASSWORD", db_cfg.get("password"))

        sql_rules = rules.get("sql", {})
        self.allowed_commands = [c.lower() for c in sql_rules.get("allowed_commands", ["select"])]
        self.forbidden_keywords = [k.lower() for k in sql_rules.get("forbidden_keywords", [])]

        if self.type != "mysql":
            raise ValueError("SqlExecutor currently only supports MySQL.")

        dsn = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        self._engine: Engine = sqlalchemy.create_engine(dsn)
        logger.info("sql_executor_init", dsn=f"{self.host}:{self.port}/{self.name}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute(self, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        """
        安全执行只读查询。
        """
        self._ensure_safe(sql)
        logger.debug("sql_execute", sql=sql, params=params)
        with self._engine.connect() as conn:
            result: Result = conn.execute(sqlalchemy.text(sql), params)
            rows = [dict(row._mapping) for row in result.fetchall()]
        logger.info("sql_rows", count=len(rows))
        return rows

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_safe(self, sql: str) -> None:
        sql_lower = sql.strip().lower()
        if not any(sql_lower.startswith(cmd) for cmd in self.allowed_commands):
            raise ValueError(f"only SELECT is allowed, got: {sql[:30]}...")
        for kw in self.forbidden_keywords:
            if kw in sql_lower:
                raise ValueError(f"forbidden keyword in SQL: {kw}")


__all__ = ["SqlExecutor"]

