from __future__ import annotations

"""
LLM 语义校验（可选模块）。

用于在规则校验 + 业务校验之后，再由 LLM 进行语义 sanity check，
例如：
  - IR 是否与原始自然语言问题明显不符；
  - SQL 结果与用户意图是否匹配；

本模块只定义接口与简单的占位实现，便于后续对接本地 Qwen2.5-7B。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticValidationResult:
    ok: bool
    reason: str = ""


class LlmSemanticValidator:
    """
    LLM 语义校验接口。

    这里不直接耦合到具体推理框架（如 vLLM / Ollama），而是预留
    一个 `llm_client` 可注入的函数式接口：

        llm_client(prompt: str) -> str

    用户在集成时可以将调用 Qwen2.5-7B 的逻辑注入进来。
    """

    def __init__(self, llm_client: Optional[callable] = None) -> None:
        self._llm_client = llm_client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def validate(
        self, original_query: str, ir: Dict[str, Any], sample_result: Any = None
    ) -> SemanticValidationResult:
        """
        :param original_query: 用户自然语言问题
        :param ir: 统一 IR（或 SQL IR）
        :param sample_result: 可选，示例检索/查询结果
        """
        if self._llm_client is None:
            # 未接入 LLM 时，默认视为通过，但记录日志
            logger.info(
                "LLM semantic validator not configured, skip semantic check."
            )
            return SemanticValidationResult(ok=True, reason="llm_not_configured")

        # 构造一个非常简洁的提示词，具体 prompt 可根据 `prompts/` 下文件增强
        prompt = (
            "你是一个用于检查查询解析是否合理的审查助手。\n"
            "请判断下面的结构化 IR 是否与用户自然语言问题语义一致。\n\n"
            f"用户问题：{original_query}\n"
            f"结构化 IR(JSON)：{ir}\n"
        )
        if sample_result is not None:
            prompt += f"\n示例结果：{sample_result}\n"

        prompt += "\n仅回答 `OK` 或 `MISMATCH`。"

        try:
            reply = str(self._llm_client(prompt)).strip().upper()
        except Exception as exc:  # pragma: no cover - 依赖外部 LLM
            logger.error("LLM semantic validation failed: %s", exc, exc_info=True)
            return SemanticValidationResult(ok=True, reason="llm_call_failed")

        if "MISMATCH" in reply:
            return SemanticValidationResult(ok=False, reason="llm_mismatch")

        return SemanticValidationResult(ok=True, reason="ok")


__all__ = ["LlmSemanticValidator", "SemanticValidationResult"]

