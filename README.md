# Hybrid Bidding QA Assistant

招标行业混合 RAG + SQL 智能问答助手。面向企业多角色（采购、销售、投融资、管理层），通过融合**结构化数据（SQL）**与**非结构化知识（RAG）**，根据问题类型动态选择检索路径，帮助用户高效发现项目机会、匹配供应商/投资人。

---

## 一、系统架构概览

- **非结构化数据**（政策文本、舆情文本）→ Embedding + 向量库（FAISS）→ RAG 语义检索  
- **结构化数据**（招标、企业、供应商、价格）→ 关系型数据库（MySQL）+ IR→SQL → 精确查询与统计  
- **半结构化**：可结构化部分存入 SQL，非结构化部分分块后做 RAG；两类数据通过统一 ID（`company_id` / `project_id`）实现跨模态联动。

### 数据与模型选型（已定）

| 用途       | 选型                     | 说明 |
|------------|--------------------------|------|
| 中枢 LLM   | Qwen2.5-7B-Instruct      | 路由 / IR 生成 / 最终回答，见 `config/model.yaml` |
| Embedding  | BAAI/bge-large-zh-v1.5   | 政策/舆情 chunk 向量化 |
| Reranker   | BAAI/bge-reranker-v2-m3  | TopK → TopN 精排 |

---

## 二、目录结构

```
Solution10/
├── README.md
├── requirements.txt
├── .env.example                 # 环境变量模板，复制为 .env 后填写
├── preprocess_main.py           # 离线预处理入口：data/raw → data/processed
│
├── config/
│   ├── model.yaml               # LLM / Embedding / Reranker 选型
│   ├── database.yaml             # MySQL 连接（支持环境变量占位）
│   ├── vector_store.yaml         # 向量库配置
│   ├── data_schema.yaml          # 向量集合 + SQL 表 + IR 结构定义
│   ├── rules.yaml                # 路由/IR/SQL/RAG/归一化/回答规则
│   └── log_config.yaml           # 日志配置
│
├── data/
│   ├── raw/                      # 原始数据（未处理）
│   │   ├── policy/               # 政策类 txt/pdf
│   │   ├── opinion/              # 舆情类 txt/pdf
│   │   └── tender/               # 招标类（可选）
│   ├── processed/                # 预处理输出（脚本自动创建）
│   │   ├── policy_chunks.json
│   │   └── opinion_chunks.json
│   └── samples/                  # 人工校验 / 调试样本
│
├── prompts/                      # 各类 prompt 模板
│   ├── router_prompt.txt
│   ├── policy_ir_prompt.txt
│   ├── opinion_ir_prompt.txt
│   ├── sql_ir_prompt.txt
│   └── answer_prompt.txt
│
├── preprocess/
│   ├── regex_extract.py          # 正则抽取：时间/金额/手机号
│   ├── normalize.py              # 时间/金额/电话标准化
│   ├── policy_chunker.py         # 政策按条/款/段切分
│   ├── opinion_chunker.py        # 舆情等长+句子边界切分
│   └── llm_ir_generator.py       # 统一 IR 生成（规则 + 可选 LLM）
│
├── validators/
│   ├── schema_validator.py       # IR schema 校验
│   ├── business_validator.py     # 业务规则校验
│   ├── confidence_checker.py     # 置信度 / 不确定性判断
│   └── llm_semantic_validator.py # 可选：LLM 语义校验
│
├── retrieval/
│   ├── embedder.py               # BGE embedding 封装
│   ├── vector_store.py           # FAISS 向量库封装
│   ├── retriever.py              # TopK 检索
│   └── reranker.py               # TopK → TopN 精排
│
├── router/
│   └── query_router.py           # RAG / SQL / Hybrid 路由决策
│
├── sql_engine/
│   ├── db_schema.py              # 从 data_schema 加载表结构
│   ├── ir2sql.py                 # IR → 安全 SELECT 映射
│   └── sql_executor.py           # 只读 SQL 执行
│
├── rag/
│   ├── context_builder.py        # RAG + SQL 上下文拼装
│   └── citation.py               # 来源引用构建
│
├── answer/
│   └── answer_generator.py       # LLM 生成最终回答
│
├── api/
│   ├── main.py                   # FastAPI 入口，完整 Query 链路
│   └── schemas.py                # 请求/响应 Pydantic 模型
│
├── logging/                      # 统一日志与 trace_id
│   ├── logger.py
│   └── trace.py
│
└── tests/
    ├── test_router.py
    ├── test_validators.py
    └── test_normalize.py
```

---

## 三、环境与依赖

### 1. 创建虚拟环境并安装依赖

```bash
cd /path/to/Solution10
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 环境变量

复制 `.env.example` 为 `.env`，按实际环境填写（用于 `config/database.yaml` 等占位符）：

```bash
cp .env.example .env
```

| 变量 | 说明 |
|------|------|
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | MySQL 连接信息，供 SQL 只读查询 |
| `VECTOR_HOST` / `VECTOR_PORT` / `VECTOR_API_KEY` | 若使用远程向量服务时可填；当前本地 FAISS 可留空 |

---

## 四、使用方式

### 方式一：离线预处理（从 data 提取并生成 chunk JSON）

将**原始 txt/pdf** 放入 `data/raw/policy/`、`data/raw/opinion/` 后，在项目根目录执行：

```bash
python preprocess_main.py
```

- 会扫描 `data/raw/policy`、`data/raw/opinion` 下所有文件（支持 `.txt`；安装 `pdfplumber` 后支持 `.pdf`）。  
- 调用 `policy_chunker` / `opinion_chunker` 生成分块。  
- 输出写入 `data/processed/policy_chunks.json`、`data/processed/opinion_chunks.json`（自动创建 `data/processed/`）。

在 PyCharm 中可直接右键运行 `preprocess_main.py`。

### 方式二：启动在线问答 API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- 开发时可加 `--reload`。  
- 接口文档：`http://<服务器IP>:8000/docs`。  
- **POST /query** 请求体示例：

```json
{
  "query": "最近有哪些招标政策？",
  "user_role": null,
  "debug": false
}
```

响应中包含 `answer`、`route`（rag/sql/hybrid）、`citations`；`debug=true` 时可看到 `trace_id`、`ir`、`conf_score` 等调试信息。

---

## 五、配置说明（简要）

| 文件 | 作用 |
|------|------|
| `config/model.yaml` | 中枢 LLM、Embedding、Reranker 模型名与参数 |
| `config/database.yaml` | MySQL 连接，支持 `${DB_HOST}` 等环境变量 |
| `config/data_schema.yaml` | 向量集合（policy_chunks / opinion_chunks）与 SQL 表结构、IR schema |
| `config/rules.yaml` | 路由阈值、RAG top_k/top_n、IR 约束、SQL 只读规则、归一化与回答兜底话术 |
| `config/log_config.yaml` | 日志格式与输出（控制台/文件） |

修改上述配置后，需重启 API 服务或重新运行预处理脚本方可生效。

---

## 六、Pipeline 流程简述

1. **Query 进入** → `router` 判断走 RAG / SQL / Hybrid。  
2. **IR 生成** → `llm_ir_generator` 产出统一 IR（规则兜底，可接 LLM），经 `schema_validator` / `business_validator` 校验。  
3. **RAG 分支** → Embedding + 向量检索 TopK → Reranker TopN → 与 SQL 结果一起交给 `context_builder` 拼上下文。  
4. **SQL 分支** → `ir2sql` 生成安全 SELECT → `sql_executor` 只读执行。  
5. **置信度** → `confidence_checker` 综合检索分与 SQL 行数，低置信时走 `answer.uncertainty_response` 兜底。  
6. **回答** → `answer_generator` 基于上下文与 IR 调用 LLM 生成最终回答，并附带 `citations`。

---

## 七、模型选型依据（摘要）

- **中枢 LLM（Qwen2.5-7B-Instruct）**：路由、IR 生成、语义理解、角色适配回答；中文与结构化输出表现好，单卡可部署，Apache 2.0。  
- **Embedding（BAAI/bge-large-zh-v1.5）**：C-MTEB 中文表现领先，适合政务/商业/招标术语，支持短查长文。  
- **Reranker（BAAI/bge-reranker-v2-m3）**：Cross-encoder 精排，复合条件判别与抗噪能力强，Apache 2.0。

---

## 八、扩展与注意

- **LLM 接入**：当前 API 使用占位 `dummy_llm_client`，实际部署时请在 `api/main.py` 中替换为对 Qwen2.5-7B（如 Ollama / vLLM / HTTP）的调用。  
- **向量索引**：若需在启动时从 `data/processed/*.json` 加载并向量化后供 RAG 使用，需在应用内增加“加载 chunks + 调用 Embedder + 写入 VectorStore/Retriever”的初始化逻辑。  
- **MySQL**：需事先按 `config/data_schema.yaml` 中 `sql_tables` 建表并导入业务数据，否则 SQL 分支无数据可查。

如遇问题，可先查看日志（`config/log_config.yaml` 中配置的路径）及 `/query` 的 `debug` 字段定位环节。

---


```bash
cd /path/to/Solution10

# 若尚未初始化
git init

# 添加所有文件（.gitignore 已排除 .env、data/raw、data/processed、logs 等）
git add .
git status   # 确认没有 .env、密码、大文件被加入

git commit -m "feat: initial Hybrid Bidding QA Assistant"

# 添加远程仓库（替换为你的 GitHub 用户名和仓库名）

# 推送（首次推送主分支）
git branch -M main
git push -u origin main
```

### 3. 注意事项

- **不要提交**：`.env`、数据库密码、API Key；`.gitignore` 已包含 `.env`。
- **不会上传**：`data/raw/`、`data/processed/`、`data/samples/`、`logs/`，避免大文件与敏感数据进仓库。
- 其他人克隆后需自行：`cp .env.example .env` 并填写、准备 MySQL 与原始数据、运行 `preprocess_main.py` 或按需加载向量。
