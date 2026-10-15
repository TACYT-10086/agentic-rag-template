"""search_policy 工具：Agentic Query-Transformed RAG 的检索入口。

流程 —— 模型自己选查询改写策略 → 检索 → 对结果打分 → 判不相关就换一个策略重试，
**总共最多 2 轮**；两轮都不相关则返回「较好的一轮」。

CLAUDE.md 的硬性要求：工具执行前必须自己校验权限，绝不依赖模型判断。
所以 search_policy() 的第一行就是 check_permission(role)，在任何检索 / LLM 调用之前。
"""

from rag.retriever import grade_documents, retrieve_many
from rag.transform import (
    DEFAULT_STRATEGY,
    STRATEGY_DESCRIPTIONS,
    available_strategies,
    resolve_strategy,
    transform,
)
from llm import chat_json

TOOL_NAME = "search_policy"

# role 由会话层（已认证的身份）注入，不放进工具的模型可见参数里 ——
# 否则就成了「让模型自己声明我是谁」，等于没有权限校验。
ALLOWED_ROLES = frozenset({"member", "coach", "admin"})

MAX_ROUNDS = 2

_SELECT_SYSTEM = (
    "你是检索策略调度器。根据用户问题的特点，从给定策略里挑一个最合适的。用 json 返回。"
)


def check_permission(role: str) -> None:
    """执行前的权限校验。不通过就抛 PermissionError，不做任何后续动作。

    这是工具的第一道门：没过门之前不选策略、不检索、不调用 LLM。
    """
    if not role or role not in ALLOWED_ROLES:
        raise PermissionError(
            f"{TOOL_NAME} 拒绝执行：角色 {role!r} 不在允许列表 {sorted(ALLOWED_ROLES)} 内"
        )


def _choose_strategy(question: str) -> str:
    """让模型从注册表里挑一个策略，返回经 resolve_strategy 校验过的规范名。"""
    options = "\n".join(
        f"- {name}：{STRATEGY_DESCRIPTIONS.get(name, '')}" for name in available_strategies()
    )

    data = chat_json(
        [
            {"role": "system", "content": _SELECT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"可选策略：\n{options}\n\n"
                    f"用户问题：{question}\n\n"
                    "请选一个策略，用 json 返回。"
                ),
            },
        ],
        schema_hint='{"strategy": "rewrite", "reason": "为什么选它"}',
    )

    # 模型可能拼错或胡诌，resolve_strategy 兜底回落默认策略
    return resolve_strategy(data.get("strategy"))


def _different_strategy(current: str) -> str:
    """返回一个与 current 不同的策略。

    确定性轮转，不再花一次 LLM 调用 —— 「换一个策略」没什么可决策的。
    """
    names = available_strategies()
    current = resolve_strategy(current)

    if current not in names:
        return names[0]
    return names[(names.index(current) + 1) % len(names)]


def _fuse(results: list[list[dict]]) -> list[dict]:
    """多路召回合并：按 id 去重，同一文档取最高分，最后按分数降序。"""
    best: dict[int, dict] = {}
    for hits in results:
        for doc in hits:
            existing = best.get(doc["id"])
            if existing is None or doc.get("score", 0.0) > existing.get("score", 0.0):
                best[doc["id"]] = doc

    return sorted(best.values(), key=lambda d: d.get("score", 0.0), reverse=True)


def _pick_best(rounds: list[dict]) -> dict:
    """两轮都不相关时挑「较好的一轮」：先比相关文档数，再比最高相似度。"""

    def score(round_: dict) -> tuple[int, float]:
        top = max((d.get("score", 0.0) for d in round_["docs"]), default=0.0)
        return (len(round_["grade"]["relevant"]), top)

    return max(rounds, key=score)


def search_policy(
    question: str,
    role: str,
    *,
    top_k: int = 3,
    strategy: str | None = None,
) -> dict:
    """Agentic 检索：选策略 → 检索 → 评分 → 不相关就换策略重试（最多 2 轮）。

    strategy 显式传入时跳过「让模型选策略」这一步，供测试与批量评测省调用、保可复现。

    返回:
      {"question", "role", "strategy_used", "sufficient",
       "docs": [最终 top_k 条],                       # 元素结构与 retrieve() 一致
       "rounds": [{"strategy", "queries", "docs", "grade"}, ...]}
    """
    # 第一行，先于一切检索与 LLM 调用
    check_permission(role)

    question = (question or "").strip()
    planned = resolve_strategy(strategy) if strategy else _choose_strategy(question)

    rounds: list[dict] = []
    for attempt in range(MAX_ROUNDS):
        name = planned if attempt == 0 else _different_strategy(rounds[0]["strategy"])

        queries = transform(question, name)
        docs = _fuse(retrieve_many(queries, top_k=top_k))[:top_k]
        grade = grade_documents(question, docs)

        rounds.append({"strategy": name, "queries": queries, "docs": docs, "grade": grade})

        if grade["sufficient"]:
            break

    best = rounds[-1] if rounds[-1]["grade"]["sufficient"] else _pick_best(rounds)

    return {
        "question": question,
        "role": role,
        "strategy_used": best["strategy"],
        "sufficient": best["grade"]["sufficient"],
        "docs": best["docs"],
        "rounds": rounds,
    }
