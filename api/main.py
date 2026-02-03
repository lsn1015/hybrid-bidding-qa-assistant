from __future__ import annotations

"""
FastAPI 入口，将 Router / IR / RAG / SQL / Answer 串成完整 pipeline。
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from ..answer.answer_generator import AnswerGenerator
from ..logging.logger import get_logger
from ..logging.trace import new_trace
from ..preprocess.llm_ir_generator import LlmIRGenerator
from ..rag.citation import build_rag_citations, build_sql_citations
from ..rag.context_builder import ContextBuilder
from ..retrieval.reranker import Reranker
from ..retrieval.retriever import Retriever
from ..router.query_router import QueryRouter
from ..sql_engine.ir2sql import IR2SQL
from ..sql_engine.sql_executor import SqlExecutor
from ..validators.confidence_checker import ConfidenceChecker, ConfidenceInput
from .schemas import QueryRequest, QueryResponse, Citation

app = FastAPI(title="Hybrid Bidding QA Assistant")

logger = get_logger(__name__)


# ---------------------------------------------------------------------- #
# 依赖初始化（简单版本：进程级单例）
# ---------------------------------------------------------------------- #

def dummy_llm_client(prompt: str) -> str:
    """
    占位 LLM 客户端。
    线上部署时，请替换为实际 Qwen2.5-7B 调用逻辑。
    """
    # 为了保证可跑通，这里给一个非常简单的提示。
    return f"[占位回答] 暂未接入真实 LLM。提示词预览：\n{prompt[:400]}"


router = QueryRouter()
ir_generator = LlmIRGenerator(llm_client=None)  # 默认仅规则 IR，后续可注入 LLM

# 这里只演示 policy_chunks 检索管道，opinion 可按需再实例化一套
policy_retriever = Retriever(collection_key="policy_chunks", top_k=20)
policy_reranker = Reranker(top_n=3)

context_builder = ContextBuilder()
ir2sql = IR2SQL()
sql_executor = SqlExecutor()
confidence_checker = ConfidenceChecker()
answer_generator = AnswerGenerator(llm_client=dummy_llm_client)


# ---------------------------------------------------------------------- #
# API
# ---------------------------------------------------------------------- #

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest) -> QueryResponse:
    # 生成 trace_id，串到整条调用链日志中
    trace_id = new_trace()
    logger.info("incoming_query", trace_id=trace_id, query=body.query)

    # 1) 路由 + IR
    route_result = router.route(body.query)
    ir_obj = ir_generator.generate(body.query)
    ir: Dict[str, Any] = ir_obj.to_dict()

    rag_docs = []
    sql_rows: List[Dict[str, Any]] = []
    sql_table: Optional[str] = None

    # 2) RAG 路径（policy 为例）
    if route_result.route in ("rag", "hybrid"):
        retrieved = policy_retriever.retrieve(body.query)
        reranked = policy_reranker.rerank(body.query, retrieved)
        rag_docs = reranked

    # 3) SQL 路径
    if route_result.route in ("sql", "hybrid"):
        sql_build = ir2sql.build(ir)
        sql_table = ir.get("table")
        sql_rows = sql_executor.execute(sql_build.sql, sql_build.params)

    # 4) 置信度评估
    conf_input = ConfidenceInput(
        retriever_scores=[d.score for d in rag_docs] if rag_docs else None,
        reranker_scores=[d.score for d in rag_docs] if rag_docs else None,
        sql_row_count=len(sql_rows) if sql_rows else None,
    )
    conf_result = confidence_checker.evaluate(conf_input)

    # 5) 构造上下文 + 引用
    context = context_builder.build(rag_docs=rag_docs, sql_rows=sql_rows)

    rag_citations = build_rag_citations(rag_docs)
    sql_citations = (
        build_sql_citations(sql_table, sql_rows) if sql_table else []
    )
    all_citations = rag_citations + sql_citations

    # 6) 生成最终回答
    answer_obj = answer_generator.generate(
        query=body.query,
        route=route_result.route,
        ir=ir,
        context=context,
        citations=all_citations,
        is_confident=conf_result.is_confident,
    )

    debug: Optional[Dict[str, Any]] = None
    if body.debug:
        debug = {
            "trace_id": trace_id,
            "route": route_result.__dict__,
            "ir": ir,
            "conf_score": conf_result.score,
        }

    return QueryResponse(
        answer=answer_obj.text,
        route=answer_obj.route,
        citations=[Citation(**c) for c in all_citations],
        debug=debug,
    )


__all__ = ["app"]

