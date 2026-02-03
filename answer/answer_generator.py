from __future__ import annotations

"""
LLM 生成最终回答。

输入：
- 用户 query；
- 路由结果 route_result；
- 统一 IR；
- RAG + SQL 拼装后的 context；
- citations；

输出：
- 结构化 Answer 对象，包含答案文本与引用信息。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

from ..logging.logger import get_logger
from ..rag.context_builder import BuiltContext

logger = get_logger(__name__)


LlmClient = Callable[[str], str]


@dataclass
class Answer:
    text: str
    citations: List[Dict[str, Any]]
    route: str
    debug: Dict[str, Any]


class AnswerGenerator:
    def __init__(self, llm_client: LlmClient) -> None:
        self._llm_client = llm_client

        project_root = Path(__file__).resolve().parents[1]
        rules_path = project_root / "config" / "rules.yaml"
        with rules_path.open("r", encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)

        prompts_dir = project_root / "prompts"
        self._answer_prompt = (prompts_dir / "answer_prompt.txt").read_text(
            encoding="utf-8"
        )

        answer_cfg = self._rules.get("answer", {})
        self.uncertainty_response: str = answer_cfg.get(
            "uncertainty_response",
            "当前信息不足以给出确定结论。",
        )

    def generate(
        self,
        query: str,
        route: str,
        ir: Dict[str, Any],
        context: BuiltContext,
        citations: List[Dict[str, Any]],
        is_confident: bool,
    ) -> Answer:
        """
        主入口：基于上下文和 IR 生成自然语言回答。
        """
        if not is_confident:
            # 低置信直接返回兜底话术，但仍附带上下文和引用，供前端展示
            return Answer(
                text=self.uncertainty_response,
                citations=citations,
                route=route,
                debug={"ir": ir, "context": context.text},
            )

        prompt = self._format_prompt(query, route, ir, context.text)
        try:
            answer_text = self._llm_client(prompt)
        except Exception as exc:  # pragma: no cover - 依赖外部 LLM
            logger.error("answer_generation_failed", error=repr(exc))
            answer_text = self.uncertainty_response

        return Answer(
            text=answer_text,
            citations=citations,
            route=route,
            debug={"ir": ir, "context": context.text},
        )

    def _format_prompt(
        self, query: str, route: str, ir: Dict[str, Any], context: str
    ) -> str:
        """
        将 answer_prompt 模板与实际变量拼接。
        """
        tmpl = self._answer_prompt
        prompt = tmpl.format(
            query=query,
            route=route,
            ir=ir,
            context=context,
        )
        return prompt


__all__ = ["AnswerGenerator", "Answer"]

