from __future__ import annotations

"""
统一日志入口（避免与标准库 logging 冲突的最小实现）。

注意：由于当前项目目录中也命名为 `logging`，容易与 Python 标准库
冲突，这里不再依赖标准库 logging 的复杂配置，而是实现一个简化版：

- 提供 `get_logger(name)`，返回带 `info / warning / error / debug` 方法的对象；
- 自动携带 trace_id（见 `logging/trace.py`）；
- 输出到 stdout，格式固定，便于在云服务器上收集日志。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .trace import get_trace


@dataclass
class SimpleLogger:
    name: str

    def _log(self, level: str, message: str, **extra: Any) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        trace_id = get_trace() or "-"
        extra_str = " ".join(f"{k}={v!r}" for k, v in extra.items()) if extra else ""
        line = f"[{ts}] [{level}] [{self.name}] [trace_id={trace_id}] {message}"
        if extra_str:
            line = f"{line} | {extra_str}"
        print(line, flush=True)

    def info(self, msg: str, **extra: Any) -> None:
        self._log("INFO", msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log("WARNING", msg, **extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log("ERROR", msg, **extra)

    def debug(self, msg: str, **extra: Any) -> None:
        self._log("DEBUG", msg, **extra)


def get_logger(name: Optional[str] = None) -> SimpleLogger:
    """
    获取一个简单 logger，name 为空时默认为 "app"。
    """
    return SimpleLogger(name=name or "app")


__all__ = ["get_logger", "SimpleLogger"]

