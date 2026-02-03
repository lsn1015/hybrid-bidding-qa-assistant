from __future__ import annotations

"""
Query Router：用户 query 的多层路由决策。

职责：
- 结合关键词 / 正则规则与 `config/rules.yaml` 中 router.intent_map；
- 判断本次查询更适合：
    - RAG（policy / opinion）；
    - SQL（tender / company / items / price 等结构化）；
    - Hybrid（同时需要语义上下文与结构化统计）。
- 可选：调用 LLM 进行辅助判定（当前接口预留，默认不用 LLM 也能工作）。
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional

import yaml

from ..logging.logger import get_logger

logger = get_logger(__name__)


class Route(str, Enum):
    RAG = "rag"
    SQL = "sql"
    HYBRID = "hybrid"
    OTHER = "other"


@dataclass
class RouteResult:
    route: str
    intent: str
    entity_type: Optional[str] = None
    reason: Optional[str] = None


LlmClassifier = Callable[[str], str]


class QueryRouter:
    """
    规则优先 + 可选 LLM 辅助的路由器。
    """

    def __init__(self, llm_classifier: Optional[LlmClassifier] = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        rules_path = project_root / "config" / "rules.yaml"

        with rules_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._router_cfg: Dict = cfg.get("router", {})
        self._llm_classifier = llm_classifier

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def route(self, query: str) -> RouteResult:
        """
        主入口：输入自然语言 query，输出 RouteResult。
        """
        # 1) 规则 + 关键词 粗分意图
        intent, entity, route = self._rule_based_route(query)

        # 2) 如配置了 LLM 分类器，可在不确定时让 LLM 微调选择
        if self._llm_classifier is not None and route == Route.OTHER.value:
            llm_label = self._safe_llm_classify(query)
            if llm_label:
                intent = llm_label
                route = self._intent_to_route(llm_label)

        logger.info(
            "query_routed",
            query=query[:50],
            intent=intent,
            entity=entity,
            route=route,
        )
        return RouteResult(route=route, intent=intent, entity_type=entity)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _rule_based_route(self, query: str) -> tuple[str, Optional[str], str]:
        q = query.lower()

        # 政策类：政策、扶持、规范、办法、通知等 -> policy_query -> rag
        if any(k in q for k in ("政策", "扶持", "规范", "办法", "通知", "条例")):
            return "policy_query", "policy", Route.RAG.value

        # 舆情类：舆情、口碑、负面、风险、投诉等 -> opinion_query -> rag
        if any(k in q for k in ("舆情", "口碑", "负面", "风险", "投诉", "舆论")):
            return "opinion_query", "opinion", Route.RAG.value

        # 招标/中标/项目 -> tender_query -> sql
        if any(k in q for k in ("招标", "中标", "投标", "项目", "标段")):
            # 若同时提到“政策/扶持”等，则认为需要 Hybrid
            if any(k in q for k in ("政策", "扶持", "补贴", "优惠")):
                return "hybrid_query", "tender", Route.HYBRID.value
            return "tender_query", "tender", Route.SQL.value

        # 公司/企业/供应商 -> company_query -> sql
        if any(k in q for k in ("公司", "企业", "供应商", "厂商")):
            return "company_query", "company", Route.SQL.value

        # 价格/行情/报价 -> items_query -> sql
        if any(k in q for k in ("价格", "报价", "行情", "单价", "成本")):
            return "items_query", "price", Route.SQL.value

        # 默认：走 RAG，一般问答
        return "general_query", None, Route.RAG.value

    def _intent_to_route(self, intent: str) -> str:
        """
        尝试根据 rules.yaml 中 router.intent_map 做 mapping。
        """
        intent_map = self._router_cfg.get("intent_map", {})
        conf = intent_map.get(intent, {})
        # 配置里既有 route 也有 router 的键，这里做兼容
        route = conf.get("route") or conf.get("router")
        if route in (Route.RAG.value, Route.SQL.value, Route.HYBRID.value):
            return route
        return Route.OTHER.value

    def _safe_llm_classify(self, query: str) -> Optional[str]:
        """
        将 query 交给外部 LLM 做高层意图分类，返回 intent label。
        """
        prompt = (
            "你是一个用于分类查询意图的助手，请根据用户问题输出以下标签之一：\n"
            "- policy_query\n"
            "- opinion_query\n"
            "- tender_query\n"
            "- company_query\n"
            "- items_query\n"
            "- hybrid_query\n"
            "只输出标签本身，不要额外解释。\n\n"
            f"用户问题：{query}"
        )
        try:
            label = str(self._llm_classifier(prompt)).strip()
            return label
        except Exception as exc:  # pragma: no cover - 依赖外部 LLM
            logger.error("llm_classifier_failed", error=repr(exc))
            return None


__all__ = ["QueryRouter", "RouteResult", "Route"]

