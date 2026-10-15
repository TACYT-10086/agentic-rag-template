"""查询改写：把用户原问题加工成更好检索的查询串。

三种策略统一契约 —— `(question) -> list[str]`，返回「待编码的查询串」而不是检索结果。
这样上层循环和评测不用按策略分支，一律拿到字符串列表交给 retriever 编码即可。
HyDE 生成的假想文档正文也走同一条路：它本身就是一段「像答案的文本」。

注意语料入库时编码的是 `标题\\n正文`（见 rag/ingest.py），而检索时编码的是裸查询串。
假想文档比短问题更贴近索引侧，这正是 HyDE 起效的原因。
"""

from collections.abc import Callable

from llm import chat, chat_json

DEFAULT_STRATEGY = "rewrite"

# 给「让模型自己选策略」的 prompt 用：策略名 -> 一句话说明 + 适用场景
STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "rewrite": "把口语化问题改写成一句检索友好的书面查询。适合表述与场馆手册用词接近的问题。",
    "multi_query": "生成多条语义相同但用词不同的查询并合并结果。适合表述模糊、一词多义的问题。",
    "hyde": "先写一段假想的答案文档，再用它去检索。适合答案措辞与问题差异大、很口语化的问题。",
}

_REWRITE_SYSTEM = (
    "你是检索查询改写器。把用户的口语化问题改写成一句简洁、书面、"
    "便于向量检索的查询，只输出改写后的查询本身，不要解释、不要加引号。"
)

_HYDE_SYSTEM = (
    "你是健身场馆的知识库写手。针对用户的问题，写一段 150-250 字的说明文字，"
    "风格模仿场馆手册，直接给出可能成立的答案细节（如时间、金额、规则）。"
    "不要用 markdown，不要复述问题，只输出正文。"
)

_MULTI_SYSTEM = (
    "你是检索查询扩写器。针对用户的问题，生成多条语义相同但用词和角度不同的检索查询，"
    "覆盖同义词、书面表达和可能的专业术语。"
)


def rewrite_query(question: str) -> list[str]:
    """把问题改写成一句检索友好的书面查询。返回长度 1 的列表。"""
    text = chat(
        [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ]
    ).strip()

    # 模型偶尔会输出空串（DeepSeek 已知问题），退回原问题至少让检索能跑
    return [text] if text else [question]


def multi_query(question: str, n: int = 3) -> list[str]:
    """生成 n 条同义异构的查询。保序去重，模型给不出东西时退回 [原问题]。"""
    data = chat_json(
        [
            {"role": "system", "content": _MULTI_SYSTEM},
            {
                "role": "user",
                "content": f"问题：{question}\n请生成 {n} 条查询，用 json 返回。",
            },
        ],
        schema_hint='{"queries": ["查询一", "查询二", "查询三"]}',
    )

    # 用 json 而不是自由文本，就是看中这里是变长列表——纯文本解析太脆
    raw = data.get("queries") or []

    seen: set[str] = set()
    queries: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            queries.append(text)

    return queries[:n] if queries else [question]


def hyde(question: str) -> list[str]:
    """生成一段假想答案文档，返回 [正文]（长度 1）。"""
    text = chat(
        [
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": question},
        ]
    ).strip()

    return [text] if text else [question]


STRATEGIES: dict[str, Callable[[str], list[str]]] = {
    "rewrite": rewrite_query,
    "multi_query": multi_query,
    "hyde": hyde,
}

# 名字归一化用：去掉大小写、空格、连字符、下划线后的映射。
# "HyDE" / "multi-query" / "multi query" 都该落到同一个策略上。
_NORMALIZED = {
    "".join(ch for ch in name.lower() if ch.isalnum()): name for name in STRATEGIES
}


def available_strategies() -> list[str]:
    """可用策略名，顺序稳定（dict 保序），供选策略 prompt 和「换一个策略」使用。"""
    return list(STRATEGIES)


def resolve_strategy(name: str | None) -> str:
    """把模型给的名字映射到注册表里的规范名。

    未知、拼错或空值一律回落 DEFAULT_STRATEGY —— 模型写错策略名绝不能弄崩循环。
    """
    key = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    return _NORMALIZED.get(key, DEFAULT_STRATEGY)


def transform(question: str, strategy: str) -> list[str]:
    """策略名 -> 待编码的查询串列表。

    策略产出为空时退回 [原问题]，保证调用方永远拿到至少一条可检索的串。
    """
    question = (question or "").strip()
    queries = STRATEGIES[resolve_strategy(strategy)](question)
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    if queries:
        return queries
    return [question] if question else []
