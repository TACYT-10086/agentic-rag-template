"""search_policy tool: entry point for Agentic Query-Transformed RAG retrieval.

Flow: the model picks a query-rewrite strategy on its own -> retrieves -> scores the
results -> if judged irrelevant, swaps to another strategy and retries, for **at most
2 rounds**; if both rounds are irrelevant, the "better round" is returned.

Hard requirement from CLAUDE.md: the tool must validate permissions itself before
executing, and must never rely on the model's judgment. Hence the first line of
search_policy() is check_permission(role), before any retrieval / LLM call.
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

# The role is injected by the session layer (an authenticated identity) and is kept out
# of the model-visible parameters of the tool -- otherwise it would amount to "letting
# the model declare who I am", which is no permission check at all.
ALLOWED_ROLES = frozenset({"member", "coach", "admin"})

MAX_ROUNDS = 2

_SELECT_SYSTEM = (
    "You are the retrieval strategy scheduler. Based on the characteristics of the "
    "user's question, pick the most suitable strategy from the given ones. "
    "Return it as json."
)


def check_permission(role: str) -> None:
    """Validate permissions before execution. Raise a PermissionError otherwise,
    performing no further action.

    This is the tool's first gate: no strategy selection, retrieval, or LLM call
    happens before passing through it.
    """
    if not role or role not in ALLOWED_ROLES:
        raise PermissionError(
            f"{TOOL_NAME} refused to run: role {role!r} is not in the allowed list "
            f"{sorted(ALLOWED_ROLES)}"
        )


def _choose_strategy(question: str) -> str:
    """Ask the model to pick a strategy from the registry and return the canonical name
    validated by resolve_strategy."""
    options = "\n".join(
        f"- {name}: {STRATEGY_DESCRIPTIONS.get(name, '')}" for name in available_strategies()
    )

    data = chat_json(
        [
            {"role": "system", "content": _SELECT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Available strategies:\n{options}\n\n"
                    f"User question: {question}\n\n"
                    "Pick one strategy and return it as json."
                ),
            },
        ],
        schema_hint='{"strategy": "rewrite", "reason": "why it was chosen"}',
    )

    # The model may misspell or make things up; resolve_strategy falls back to the
    # default strategy as a safety net
    return resolve_strategy(data.get("strategy"))


def _different_strategy(current: str) -> str:
    """Return a strategy different from current.

    Deterministic rotation, no extra LLM call -- "try another strategy" requires no
    decision-making.
    """
    names = available_strategies()
    current = resolve_strategy(current)

    if current not in names:
        return names[0]
    return names[(names.index(current) + 1) % len(names)]


def _fuse(results: list[list[dict]]) -> list[dict]:
    """Merge multi-way recall: deduplicate by id, keep the highest score per document,
    and finally sort in descending score order."""
    best: dict[int, dict] = {}
    for hits in results:
        for doc in hits:
            existing = best.get(doc["id"])
            if existing is None or doc.get("score", 0.0) > existing.get("score", 0.0):
                best[doc["id"]] = doc

    return sorted(best.values(), key=lambda d: d.get("score", 0.0), reverse=True)


def _pick_best(rounds: list[dict]) -> dict:
    """When both rounds are irrelevant, pick the "better round": first compare the
    number of relevant documents, then the highest similarity."""

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
    """Agentic retrieval: pick a strategy -> retrieve -> grade -> if irrelevant, retry
    with another strategy (at most 2 rounds).

    When strategy is passed explicitly, the "let the model pick a strategy" step is
    skipped, which saves calls and keeps tests / batch eval reproducible.

    Returns:
      {"question", "role", "strategy_used", "sufficient",
       "docs": [final top_k items],                   # element structure matches retrieve()
       "rounds": [{"strategy", "queries", "docs", "grade"}, ...]}
    """
    # First line, before every retrieval and LLM call
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