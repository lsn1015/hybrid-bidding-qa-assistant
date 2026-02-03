from __future__ import annotations

"""
LLM 生成统一 IR（intermediate representation）中间表示。

设计要点：
- IR 结构参考 `config/data_schema.yaml` 中 `unified_query_ir`；
- 校验依赖 `validators.schema_validator` 与 `validators.business_validator`；
- 人工审核仅在低置信或异常场景触发（由上层逻辑决定）。
"""

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

from ..logging.logger import get_logger
from ..router.query_router import QueryRouter, RouteResult
from ..validators.business_validator import BusinessValidator
from ..validators.schema_validator import IRSchemaValidator

logger = get_logger(__name__)


LlmClient = Callable[[str], str]


@dataclass
class UnifiedQueryIR:
    """
    与 data_schema 中 unified_query_ir 对齐的 Python 表示。
    """

    query_type: str
    entity_type: str
    fallback_enabled: bool
    original_query: str
    structured_conditions: Dict[str, Any]
    unstructured_keywords: List[str]

    # SQL 相关字段（参考 rules.ir.required_fields）
    table: Optional[str] = None
    filters: Optional[List[Dict[str, Any]]] = None
    select: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LlmIRGenerator:
    """
    IR 生成入口。

    - 优先使用规则 + Router 进行粗分类（query_type / entity_type）；
    - 若注入了 LLM，则构造 prompt 要求输出 JSON IR，并做 schema + 业务校验；
    - 若未注入 LLM，则使用简化规则生成一个“保底 IR”，保证系统可跑通。
    """

    def __init__(self, llm_client: Optional[LlmClient] = None) -> None:
        self._llm_client = llm_client
        self._router = QueryRouter()
        self._schema_validator = IRSchemaValidator()
        self._business_validator = BusinessValidator()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, query: str) -> UnifiedQueryIR:
        """
        主入口：输入自然语言 query，输出结构化 IR。
        """
        route_result = self._router.route(query)

        if self._llm_client is None:
            ir = self._build_rule_based_ir(query, route_result)
        else:
            ir = self._build_llm_ir(query, route_result)

        # 统一做 schema + 业务校验（失败时抛异常或记录日志）
        schema_result = self._schema_validator.validate(ir)
        if not schema_result.ok:
            logger.warning("IR schema validation failed", errors=schema_result.errors)

        biz_result = self._business_validator.validate_ir(ir)
        if not biz_result.ok:
            logger.info("IR business warnings", warnings=biz_result.warnings)

        return UnifiedQueryIR(**ir)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_rule_based_ir(self, query: str, route_result: RouteResult) -> Dict[str, Any]:
        """
        无 LLM 时的兜底 IR 生成逻辑：
        - 根据路由结果确定 query_type / entity_type；
        - 结构化条件只做非常有限的 keyword 提取，主要为后续 SQL / RAG 提供 hint。
        """
        qt = route_result.route
        entity = route_result.entity_type or "unknown"

        structured: Dict[str, Any] = {}
        # 非严格：简单关键字 hint
        if "公司" in query or "企业" in query:
            structured["company_name"] = query
        if "项目" in query or "招标" in query or "中标" in query:
            structured["project_name"] = query

        unstructured_keywords = [w for w in query.replace("，", " ").split() if w]

        ir: Dict[str, Any] = {
            "query_type": qt,
            "entity_type": entity,
            "fallback_enabled": True,
            "original_query": query,
            "structured_conditions": structured,
            "unstructured_keywords": unstructured_keywords,
        }

        # 若推断为 SQL 查询，提供最基础的 SQL IR 字段
        if qt in ("sql", "hybrid"):
            ir.setdefault("table", "tender_project")
            ir.setdefault("select", ["*"])
            ir.setdefault("filters", [])

        logger.info("build_rule_based_ir", route=qt, entity=entity)
        return ir

    def _build_llm_ir(self, query: str, route_result: RouteResult) -> Dict[str, Any]:
        """
        调用 LLM 生成 IR 的路径：
        - 构造紧凑 prompt；
        - 要求 LLM 严格输出 JSON；
        - 若解析失败则退回规则路径。
        """
        prompt = (
            "你是招投标行业智能助手的解析模块，请将用户问题转换为统一的 JSON IR。\n"
            "字段要求：\n"
            "- query_type: 'rag' | 'sql' | 'hybrid'\n"
            "- entity_type: 'policy' | 'opinion' | 'company' | 'tender' | 'material' | 'price'\n"
            "- fallback_enabled: bool\n"
            "- original_query: str\n"
            "- structured_conditions: 对公司/项目/金额/时间等显式条件进行结构化\n"
            "- unstructured_keywords: 用于 RAG 的关键词列表\n"
            "若 query_type 属于 'sql' 或 'hybrid'，还需补充：\n"
            "- table: 目标主表，如 'tender_project'、'company_master' 等\n"
            "- select: 需要返回的字段列表\n"
            "- filters: 由 {field, op, value} 组成的列表，op 只允许：=, >, <, >=, <=, between, like\n\n"
            f"用户问题：{query}\n"
            f"Router 预判：query_type={route_result.route}, entity_type={route_result.entity_type}\n"
            "请直接输出 JSON，不要包含额外说明。"
        )

        try:
            raw = self._llm_client(prompt)  # type: ignore[operator]
            import json

            ir = json.loads(raw)
            logger.info("build_llm_ir_success", route=ir.get("query_type"))
            return ir
        except Exception as exc:  # pragma: no cover - 依赖外部 LLM
            logger.error("build_llm_ir_failed, fallback to rule-based", error=repr(exc))
            return self._build_rule_based_ir(query, route_result)


__all__ = ["UnifiedQueryIR", "LlmIRGenerator"]

