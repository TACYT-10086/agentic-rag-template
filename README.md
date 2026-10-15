# Agentic RAG Core

Agentic RAG 的核心逻辑模板。提供查询改写、检索、结果评分与策略重试循环。

## 核心方法

### transform.py — 查询改写
- `transform(question, strategy) -> list[str]`：策略入口，统一获取待检索的查询串。
- `rewrite_query(question) -> list[str]`：口语化问题转为书面检索式。
- `multi_query(question, n) -> list[str]`：生成同义异构的 n 条查询，保序去重。
- `hyde(question) -> list[str]`：生成假想答案文档，适应索引侧“标题\n正文”结构。
- `resolve_strategy(name) -> str`：策略名归一化，未知输入回落默认策略。

### retriever.py — 检索与评分
- `retrieve(query, top_k) -> list[dict]`：单查询检索，返回文档及余弦相似度。
- `retrieve_many(queries, top_k) -> list[list[dict]]`：批量编码多路查询，单次加载集合。
- `grade_documents(question, docs, threshold) -> dict`：LLM 批量判定相关性，返回 `relevant`、`irrelevant`、`grades`、`sufficient`。

### search_policy.py — 决策循环
- `search_policy(question, role, top_k, strategy) -> dict`：Agentic 检索主入口，最多 2 轮策略重试，返回最终文档与重试轮次记录。
- `check_permission(role) -> None`：硬性权限校验，第一行执行，未通过直接抛出 `PermissionError`。

## 设计要点
- **权限前置**：`check_permission` 严格先于一切 LLM 与向量检索。
- **本地兜底**：模型乱序或缺失 `index` 时，`grade_documents` 采取 fail-open 按相关处理；`sufficient` 由本地计数决定。
- **零冗余调用**：换策略采用本地确定性轮转，不额外消耗 LLM。
- **类型防御**：显式排除 `bool` 被当 `int` 索引的隐患（`True/False` 转为 `0/1`）。
- **多路融合**：`_fuse` 按 `id` 去重取最高分；`_pick_best` 按“相关数+最高相似度”返回较优轮次。