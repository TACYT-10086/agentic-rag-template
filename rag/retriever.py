"""向量检索 + 检索结果评分。

retrieve() 是最朴素的基线：原问题不加工，直接编码后查 top-k。
grade_documents() 是给 Agentic RAG 用的：用 DeepSeek 一次性判定检索结果是否切题。
"""

from config import require_env
from llm import chat_json
from rag.embedding import embed
from rag.vector_store import get_milvus_client

# 送进评分 prompt 的正文长度上限。评分只看切不切题，没必要把全文塞进去烧 token。
_CONTENT_PREVIEW = 300

_GRADE_SYSTEM = (
    "你是检索结果相关度评审。逐个判断候选文档能否帮助回答用户问题："
    "只要文档内容与问题主题相关就算相关，不必直接给出答案。"
    "不要臆测文档之外的信息，逐条给出判断理由。用 json 返回。"
)


def _hit_to_doc(hit: dict) -> dict:
    return {
        "id": hit["id"],
        "title": hit["entity"]["title"],
        "content": hit["entity"]["content"],
        "category": hit["entity"]["category"],
        "source": hit["entity"]["source"],
        "score": hit["distance"],
    }


def _truncate(text: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= _CONTENT_PREVIEW:
        return text
    return text[:_CONTENT_PREVIEW] + "…"


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """检索与 query 最相关的 top_k 条知识文档。

    返回 [{"id","title","content","category","source","score"}, ...]，
    score 是余弦相似度，越大越相关。
    """
    db_name = require_env("DB_NAME")
    collection = require_env("COLLECTION_NAME")
    client = get_milvus_client(db_name=db_name)

    # 幂等；避免集合未加载时检索报错
    client.load_collection(collection)

    hits = client.search(
        collection_name=collection,
        data=[embed([query])[0]],
        limit=top_k,
        output_fields=["title", "content", "category", "source"],
    )

    return [_hit_to_doc(hit) for hit in hits[0]]


def retrieve_many(queries: list[str], top_k: int = 3) -> list[list[dict]]:
    """一次编码多条查询、只加载一次集合，逐条检索。

    multi_query 一次能产出多条查询，若逐条走 retrieve()，每条都要重建 Milvus
    客户端 + load_collection，纯属浪费。返回结果与 queries 一一对应。
    """
    if not queries:
        return []

    db_name = require_env("DB_NAME")
    collection = require_env("COLLECTION_NAME")
    client = get_milvus_client(db_name=db_name)
    client.load_collection(collection)

    hits_per_query = client.search(
        collection_name=collection,
        data=embed(queries),
        limit=top_k,
        output_fields=["title", "content", "category", "source"],
    )

    return [[_hit_to_doc(hit) for hit in hits] for hits in hits_per_query]


def grade_documents(question: str, docs: list[dict], *, threshold: int = 1) -> dict:
    """用一次 LLM 调用批量判定每个 doc 是否与问题相关。

    返回:
      {"relevant": [...], "irrelevant": [...],
       "grades": [bool, ...],     # 与入参 docs 等长同序，调用方可按下标对齐
       "sufficient": bool,        # 相关文档数 >= threshold
       "reason": str}             # 模型的整体说明，可能为空

    结果按 index 回填而非按返回顺序，模型乱序返回也不会错配。
    个别 index 缺失按相关处理（fail-open）：为了一个缺项就重试一轮不划算。
    """
    if not docs:
        return {
            "relevant": [],
            "irrelevant": [],
            "grades": [],
            "sufficient": False,
            "reason": "检索结果为空，无需评分",
        }

    listing = "\n\n".join(
        f"[{i}] 标题：{doc.get('title', '')}\n正文：{_truncate(doc.get('content', ''))}"
        for i, doc in enumerate(docs)
    )

    data = chat_json(
        [
            {"role": "system", "content": _GRADE_SYSTEM},
            {
                "role": "user",
                "content": f"用户问题：{question}\n\n候选文档：\n{listing}",
            },
        ],
        schema_hint=(
            '{"grades": [{"index": 0, "relevant": true, "reason": "为什么相关"}],'
            ' "sufficient": true}'
        ),
    )

    raw_grades = data.get("grades") or []
    reason = raw_grades[0].get("reason") or ""

    verdicts: dict[int, bool] = {}
    for item in raw_grades:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        # bool 是 int 的子类，得排掉，否则 True/False 会被当成 0/1 索引
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if 0 <= index < len(docs):
            verdicts[index] = bool(item.get("relevant"))

    grades = [verdicts.get(i, True) for i in range(len(docs))]

    relevant = [doc for doc, ok in zip(docs, grades) if ok]
    irrelevant = [doc for doc, ok in zip(docs, grades) if not ok]

    return {
        "relevant": relevant,
        "irrelevant": irrelevant,
        "grades": grades,
        # 模型自报的 sufficient 不可信（它不知道 threshold 是多少），以本地计数为准
        "sufficient": len(relevant) >= threshold,
        "reason": str(reason or ""),
    }
